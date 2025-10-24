from typing import Any, Optional, Iterable

import numpy as np

from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.data.util.select_units import select_units, SelectionSet, SelectionTarget


def _stand_metrics(trees: ReferenceTrees) -> dict[str, float]:
    stems = float(np.nansum(trees.stems_per_ha)) if trees.size > 0 else 0.0
    mean_dbh = float(np.nanmean(trees.breast_height_diameter)) if trees.size > 0 else float("nan")
    mean_h = float(np.nanmean(trees.height)) if trees.size > 0 else float("nan")
    return {"stems_per_ha_total": stems, "mean_dbh": mean_dbh, "mean_h": mean_h}


def _snapshot_removed(trees: ReferenceTrees, removed_f: np.ndarray) -> dict[str, Any]:
    sel = removed_f > 0
    idx = np.where(sel)[0]
    rows: list[dict[str, Any]] = []
    for i in idx:
        rows.append({
            "identifier": str(trees.identifier[i]),
            "species": int(trees.species[i]) if not np.isnan(trees.species[i]) else -1,
            "breast_height_diameter": float(trees.breast_height_diameter[i]),
            "height": float(trees.height[i]),
            "stems_per_ha_removed": float(removed_f[i]),
        })
    return {
        "count": int(idx.size),
        "removed_stems_per_ha_sum": float(np.nansum(removed_f[sel])) if idx.size else 0.0,
        "rows": rows,
    }


def _build_filter_sfunction(
    species: Optional[Iterable[int]] = None,
    dbh_min: Optional[float] = None,
    dbh_max: Optional[float] = None,
    h_min: Optional[float] = None,
    h_max: Optional[float] = None,
):
    """Return an sfunction(data) -> bool[ndata] combining simple filters."""
    # Pre-normalize species to a set for fast membership checks
    spp_set = set(species) if species is not None else None

    def sfunction(_ctx: ForestStand, data: ReferenceTrees) -> np.ndarray:
        n = data.size
        if n == 0:
            return np.zeros(0, dtype=bool)
        mask = np.ones(n, dtype=bool)

        if spp_set is not None:
            # note: species coded as int in SoA; guard NaNs as non-members
            spv = data.species
            in_spp = np.array([(int(s) in spp_set) if not np.isnan(s) else False for s in spv], dtype=bool)
            mask &= in_spp

        if dbh_min is not None:
            mask &= np.nan_to_num(data.breast_height_diameter, nan=np.inf) >= dbh_min
        if dbh_max is not None:
            mask &= np.nan_to_num(data.breast_height_diameter, nan=-np.inf) <= dbh_max
        if h_min is not None:
            mask &= np.nan_to_num(data.height, nan=np.inf) >= h_min
        if h_max is not None:
            mask &= np.nan_to_num(data.height, nan=-np.inf) <= h_max

        return mask

    return sfunction


def cutting(input_: OpTuple[ForestStand], /, **operation_parameters) -> OpTuple[ForestStand]:
    """
    Generalized harvesting/thinning/ajourat treatment.

    Accepts a rich parameter set, then:
      1) Builds selection sets (or uses provided 'sets').
      2) Runs select_units to compute removals.
      3) Subtracts removals from trees.stems_per_ha.
      4) Records removed trees + summary in cdata.

    Key parameters (all via **operation_parameters):

    Basic target (global):
      - target_type: str | None  # 'relative' | 'absolute' | 'absolute_remain' | None
      - target_var: str | None   # e.g., 'stems_per_ha' (default)
      - target_amount: float | None

    Set construction (if 'sets' is omitted, we build 1 default set):
      - proportion: float in [0,1]            # relative amount for default set (like ajourat)
      - order_var: str                         # default 'breast_height_diameter'
      - profile: str                           # 'flat'|'below'|'above'|'even' (alias of 'flat')
      - profile_x/profile_y/profile_xmode      # for custom profile; overrides 'profile' if provided
      - species: Iterable[int]                 # filter to species (optional)
      - dbh_min/dbh_max, h_min/h_max           # numeric filters (optional)

    Misc:
      - freq_var: str                          # default 'stems_per_ha'
      - select_from_all: bool                  # default False
      - mode: str                              # default 'odds_units'
      - labels: list[str]                      # stored in cdata
      - sim_time: int                          # stored in cdata
      - sets: list[SelectionSet-like dict]     # advanced: pass fully-defined sets to bypass builder
      - Target: dict                           # advanced: {type, var, amount} for SelectionTarget

    Returns updated (stand, cdata).
    """
    stand, cdata = input_

    # --- guards
    trees: Optional[ReferenceTrees] = getattr(stand, "reference_trees", None)
    if trees is None or not isinstance(trees, ReferenceTrees):
        raise MetsiException("cutting requires vectorized ReferenceTrees: stand.reference_trees (SoA).")

    # --- common params / defaults
    freq_var: str = operation_parameters.get("freq_var", "stems_per_ha")
    select_from_all: bool = bool(operation_parameters.get("select_from_all", False))
    mode: str = operation_parameters.get("mode", "odds_units")

    labels: Optional[list[str]] = operation_parameters.get("labels")
    sim_time: Optional[int] = operation_parameters.get("sim_time")

    # --- Target (global)
    target_dict: Optional[dict[str, Any]] = operation_parameters.get("Target")
    if target_dict is None:
        # If omitted, supply an empty SelectionTarget (type-safe; linter-friendly)
        target_decl = SelectionTarget()

        target_decl.type = None # type: ignore
        target_decl.var = None # type: ignore
        target_decl.amount = None # type: ignore
    else:
        target_decl = SelectionTarget()
        target_decl.type=target_dict.get("type")
        target_decl.var=target_dict.get("var")
        target_decl.amount=target_dict.get("amount")


    # --- Build sets
    sets_param = operation_parameters.get("sets")
    sets_py: list[SelectionSet[ForestStand, ReferenceTrees]] = []

    if sets_param is not None:
        # Advanced path: user provided complete sets
        for s in sets_param:
            ss = SelectionSet[ForestStand, ReferenceTrees]()
            ss.sfunction = s["sfunction"]
            ss.order_var = s["order_var"]
            ss.target_var = s["target_var"]
            ss.target_type = s["target_type"]
            ss.target_amount = s["target_amount"]
            ss.profile_x = np.asarray(s["profile_x"], dtype=np.float64)
            ss.profile_y = np.asarray(s["profile_y"], dtype=np.float64)
            ss.profile_xmode = s["profile_xmode"]
            ss.profile_xscale = s.get("profile_xscale")
            sets_py.append(ss)
    else:
        # Default set path: build from simple parameters
        proportion = operation_parameters.get("proportion")
        if proportion is None:
            raise MetsiException("cutting requires 'proportion' (0..1) when 'sets' is not provided.")
        proportion = float(proportion)
        if not 0.0 <= proportion <= 1.0:
            raise MetsiException("'proportion' must be in [0, 1].")

        order_var = operation_parameters.get("order_var", "breast_height_diameter")
        profile = operation_parameters.get("profile", "flat")

        # User-provided custom profile arrays override 'profile' shortcut
        profile_x = operation_parameters.get("profile_x")
        profile_y = operation_parameters.get("profile_y")
        profile_xmode = operation_parameters.get("profile_xmode", "relative")

        if profile_x is None or profile_y is None:
            # build from shortcut keyword
            if profile in ("flat", "even"):
                px, py = (np.array([0.0, 1.0]), np.array([0.5, 0.5]))
            elif profile == "below":
                # thin from below: favor small end -> high prob at x=0, low at x=1
                px, py = (np.array([0.0, 1.0]), np.array([1.0, 0.0]))
            elif profile == "above":
                # thin from above: favor large end -> low prob at x=0, high at x=1
                px, py = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
            else:
                raise MetsiException(f"Unknown profile '{profile}'. Use flat/below/above or provide profile_x/y.")
        else:
            px = np.asarray(profile_x, dtype=np.float64)
            py = np.asarray(profile_y, dtype=np.float64)
            if px.shape != py.shape or px.ndim != 1 or px.size < 2:
                raise MetsiException("profile_x/profile_y must be 1D arrays of equal length (>=2).")

        # Optional filters (species, dbh/height ranges)
        sfunc = _build_filter_sfunction(
            species=operation_parameters.get("species"),
            dbh_min=operation_parameters.get("dbh_min"),
            dbh_max=operation_parameters.get("dbh_max"),
            h_min=operation_parameters.get("h_min"),
            h_max=operation_parameters.get("h_max"),
        )

        ss = SelectionSet[ForestStand, ReferenceTrees]()
        ss.sfunction = sfunc
        ss.order_var = order_var
        ss.target_var = operation_parameters.get("target_var", "stems_per_ha")
        ss.target_type = operation_parameters.get("target_type", "relative")
        ss.target_amount = proportion
        ss.profile_x = px
        ss.profile_y = py
        ss.profile_xmode = profile_xmode
        ss.profile_xscale = operation_parameters.get("profile_xscale")
        sets_py.append(ss)

    # --- Run selection
    metrics_before = _stand_metrics(trees)

    removed_f = select_units(
        context=stand,
        data=trees,
        target_decl=target_decl,
        sets=sets_py,
        freq_var=freq_var,
        select_from_all=select_from_all,
        mode=mode,
    )

    if np.any(removed_f < 0):
        raise MetsiException("cutting produced negative removals; check target/profile configuration.")

    # Apply removals (clamped to available)
    removed_f = np.minimum(removed_f, trees.stems_per_ha)
    if not trees.stems_per_ha.flags.writeable:
        trees.stems_per_ha = trees.stems_per_ha.copy()
    trees.stems_per_ha -= removed_f

    metrics_after = _stand_metrics(trees)

    # --- Collected data
    cdata.extend_list_result("removed_trees", [_snapshot_removed(trees, removed_f)])
    cdata.store("cutting", {
        "time": sim_time,
        "labels": labels or [],
        "freq_var": freq_var,
        "select_from_all": select_from_all,
        "mode": mode,
        "target": {"type": target_decl.type, "var": target_decl.var, "amount": target_decl.amount},
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "removed_stems_per_ha": float(np.nansum(removed_f)),
    })

    return (stand, cdata)
