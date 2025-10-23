from typing import Any, Optional

import numpy as np

from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.data.util.select_units import select_units, SelectionSet, SelectionTarget

def _removed_snapshot(trees: ReferenceTrees, removed_f: np.ndarray) -> dict[str, Any]:
    """
    Compact snapshot of removed trees (rows with removed_f > 0), including the removed stems
    and a few useful attributes for downstream reporting.
    """
    sel = removed_f > 0
    if not np.any(sel):
        return {
            "count": 0,
            "removed_stems_per_ha_sum": 0.0,
            "rows": [],
        }

    idx = np.where(sel)[0]
    rows = []
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
        "removed_stems_per_ha_sum": float(np.nansum(removed_f[sel])),
        "rows": rows,
    }

def _compute_stand_metrics(trees: ReferenceTrees) -> dict[str, float]:
    """ Very light metrics for before/after snapshots. """
    stems = np.nansum(trees.stems_per_ha) if trees.size > 0 else 0.0
    mean_d = float(np.nanmean(trees.breast_height_diameter)) if trees.size > 0 else float("nan")
    mean_h = float(np.nanmean(trees.height)) if trees.size > 0 else float("nan")
    return {"stems_per_ha_total": float(stems), "mean_dbh": mean_d, "mean_h": mean_h}

def ajourat(input_: OpTuple[ForestStand], /, **operation_parameters) -> OpTuple[ForestStand]:
    """
    Ajourat thinning (vectorized / SoA only).

    Parameters via **operation_parameters:
      - proportion: float  (required)  e.g. 0.3  -> share of stems to remove by profile
      - tree_selection: Optional[dict]           custom selection config (Target+sets)
      - labels: Optional[list[str]]              stored in cdata
      - sim_time: Optional[int]                  stored in cdata

    Behavior (matches R trt_ajourat defaults):
      - If no custom tree_selection is given:
          * global Target is unused (amount NA)
          * one selection set:
              sfunction: all TRUE,
              order_var: breast_height_diameter (R: "d"),
              target_var: stems_per_ha (R: "f"),
              target_type: "relative",
              target_amount: proportion,
              profile: flat [ (0.0,0.5), (1.0,0.5) ], xmode="relative"
      - Applies removals to trees.stems_per_ha.
      - Adds removed trees snapshot and summary to CollectedData.
    """
    stand, cdata = input_

    # --- Required parameter
    if "proportion" not in operation_parameters:
        raise MetsiException("ajourat requires parameter 'proportion' (0..1)")
    proportion = float(operation_parameters["proportion"])
    if not 0.0 <= proportion <= 1.0:
        raise MetsiException("ajourat 'proportion' must be within [0, 1]")

    # Optional passthroughs for reporting
    labels: Optional[list[str]] = operation_parameters.get("labels")
    sim_time: Optional[int] = operation_parameters.get("sim_time")

    # --- SoA-only guard
    trees: Optional[ReferenceTrees] = getattr(stand, "reference_trees", None)
    if trees is None or not isinstance(trees, ReferenceTrees):
        raise MetsiException("ajourat requires vectorized trees in stand.reference_trees (SoA).")

    metrics_before = _compute_stand_metrics(trees)

    # --- Build default selection if not provided (mirror R defaults)
    tree_selection: Optional[dict[str, Any]] = operation_parameters.get("tree_selection")
    if tree_selection is None:
        tree_selection = {
            # R defines Target as 'absolute_remain' with amount NA; effectively unused here.
            "Target": {"type": None, "var": None, "amount": None},
            "sets": [
                {
                    # Eligible set: all trees
                    "sfunction": lambda ctx, data: np.ones(data.size, dtype=bool),
                    # Order variable: diameter at breast height (R "d")
                    "order_var": "breast_height_diameter",
                    # Target variable (and freq var): stems per hectare (R "f")
                    "target_var": "stems_per_ha",
                    "target_type": "relative",
                    "target_amount": proportion,
                    # Flat profile over the order variable
                    "profile_x": np.array([0.0, 1.0]),
                    "profile_y": np.array([0.5, 0.5]),
                    "profile_xmode": "relative",
                    # optional: "profile_xscale": None
                }
            ],
        }  # :contentReference[oaicite:2]{index=2}

    # --- Translate selection sets for the selector
    sets_py: list[SelectionSet[ForestStand, ReferenceTrees]] = []
    for s in tree_selection.get("sets", []):
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

    # --- Run selection (no global Target => pass None), select from remaining stems
    removed_f = select_units(
        context=stand,
        data=trees,
        target_decl=SelectionTarget(type=None, var=None, amount=None),
        sets=sets_py,
        freq_var="stems_per_ha",
        select_from_all=False,      # select from remaining amount as sets accumulate
        mode="odds_units",
    )  # :contentReference[oaicite:3]{index=3}

    if np.any(removed_f < 0):
        raise MetsiException("ajourat produced negative removals; check selection profile/targets.")

    # Clamp to available stems to be safe and apply
    removed_f = np.minimum(removed_f, trees.stems_per_ha)
    if not trees.stems_per_ha.flags.writeable:
        trees.stems_per_ha = trees.stems_per_ha.copy()
    trees.stems_per_ha -= removed_f

    metrics_after = _compute_stand_metrics(trees)

    # --- Collected data (removed trees + summary)
    cdata.extend_list_result("removed_trees", [_removed_snapshot(trees, removed_f)])
    cdata.store("ajourat", {
        "time": sim_time,
        "labels": labels or [],
        "proportion": proportion,
        "removed_stems_per_ha": float(np.nansum(removed_f)),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    })

    return (stand, cdata)
