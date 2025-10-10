from __future__ import annotations
from typing import Any, Optional
import numpy as np
import numpy.typing as npt

from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.domain.forestry_operations.metrics.stand_metrics import compute_stand_metrics
from lukefi.metsi.domain.forestry_operations.metrics.selection_data import SelectionData
from lukefi.metsi.data.util.select_units import SelectionSet, SelectionTarget, select_units
from lukefi.metsi.data.limit_tables.limit_constants import (
    basal_area_instruction_lower_limit,
    basal_area_instructions_upper_limit,
)

def _removed_snapshot(trees: ReferenceTrees, removed_f: npt.NDArray[np.float64]) -> dict[str, npt.NDArray]:
    mask = removed_f > 0
    if not np.any(mask):
        return {"stems_per_ha": np.zeros(0, dtype=np.float64)}
    return {
        "stems_per_ha": removed_f[mask],
        "breast_height_diameter": trees.breast_height_diameter[mask],
        "height": getattr(trees, "height", None)[mask] if hasattr(trees, "height") else None,
        "species": getattr(trees, "species", None)[mask] if hasattr(trees, "species") else None,
    }

def ftrt_thin_basal_area(op: OpTuple[ForestStand], *, 
                         ba_after: Optional[float] = None,
                         max_proportion: Optional[float] = None,
                         tree_selection: Optional[dict[str, Any]] = None,
                         labels: Optional[list[str]] = None,
                         sim_time: Optional[int] = None) -> OpTuple[ForestStand]:

    """
    Basal-area thinning (generic core).
    - If ba_after is given, global target := max( (G - ba_after)/G, 0 ), optionally capped by max_proportion.
    - If only max_proportion is given, global target := max_proportion (relative on 'g').
    - If neither given, derive ba_after from instruction lower limit (TEMP constants shim).
    - 'tree_selection' may override sets; if None, we default to diameter-ordered from-below thinning.
    """
    stand, cdata = op

    trees: Optional[ReferenceTrees] = getattr(stand, "reference_trees", None)
    if trees is None or not isinstance(trees, ReferenceTrees):
        raise MetsiException("thin_basal_area requires vectorized trees in 'reference_trees'.")

    metrics_before = compute_stand_metrics(trees)
    G = metrics_before["G"]

    # Nothing to thin → no-op
    if not np.isfinite(G) or G <= 0:
        cdata.store("thin_basal_area", {
            "time": sim_time,
            "labels": (labels or []) + ["thinning", "cutting"],
            "target_relative_g": 0.0,
            "removed_stems_per_ha": 0.0,
            "removed_basal_area": 0.0,
            "metrics_before": metrics_before,
            "metrics_after": metrics_before,
            "G_instruction_upper": basal_area_instructions_upper_limit(stand),
            "G_instruction_lower": basal_area_instruction_lower_limit(stand),
        })
        return (stand, cdata)

    sel_data = SelectionData(trees)

    # --- 1) Build a canonical selection (target_decl + sets_py) ---
    def _compute_rel_target_from_caps(G: float) -> float:
        # derive ba_after if not given
        _ba_after = basal_area_instruction_lower_limit(stand) if ba_after is None else ba_after
        rel = 0.0
        if np.isfinite(_ba_after):
            rel = max(0.0, min(1.0, (G - _ba_after) / G))
        if max_proportion is not None:
            rel = min(rel, max_proportion)
        return float(rel)

    def _make_default_selection(rel: float) -> tuple[SelectionTarget, list[SelectionSet[ForestStand, SelectionData]]]:
        td = SelectionTarget()
        td.type = "relative"
        td.var = "g"
        td.amount = rel

        ss = SelectionSet[ForestStand, SelectionData]()
        ss.sfunction = lambda ctx, data: np.ones(data.size, dtype=bool)
        ss.order_var = "breast_height_diameter"
        ss.target_var = "g"
        ss.target_type = "relative"
        ss.target_amount = 1.0
        ss.profile_x = np.array([0.0, 1.0], dtype=np.float64)
        ss.profile_y = np.array([1.0, 0.0], dtype=np.float64)  # from-below
        ss.profile_xmode = "relative"
        ss.profile_xscale = None
        return td, [ss]

    def _map_var(name: str) -> str:
        # map short aliases to vectorized field names
        return {"g": "g", "f": "stems_per_ha", "d": "breast_height_diameter", "v": "v"}.get(name, name)

    if tree_selection is None:
        rel_target = _compute_rel_target_from_caps(G)
        target_decl, sets_py = _make_default_selection(rel_target)
        log_rel_target = rel_target  # for cdata logging
    else:
        # use provided selection; also extract rel target on g for logging if present
        target = tree_selection.get("Target", {}) or {}
        target_decl = SelectionTarget()
        target_decl.type = target.get("type")
        target_decl.var = _map_var(target.get("var"))
        target_decl.amount = target.get("amount")

        sets_py: list[SelectionSet[ForestStand, SelectionData]] = []
        for s in tree_selection.get("sets", []):
            ss = SelectionSet[ForestStand, SelectionData]()
            ss.sfunction = s["sfunction"]
            ss.order_var = _map_var(s["order_var"])
            ss.target_var = _map_var(s["target_var"])
            ss.target_type = s["target_type"]
            ss.target_amount = s["target_amount"]
            ss.profile_x = np.asarray(s["profile_x"], dtype=np.float64)
            ss.profile_y = np.asarray(s["profile_y"], dtype=np.float64)
            ss.profile_xmode = s["profile_xmode"]
            ss.profile_xscale = s.get("profile_xscale")
            sets_py.append(ss)

        log_rel_target = (
            float(target_decl.amount)
            if (target_decl.type == "relative" and target_decl.var == "g" and target_decl.amount is not None)
            else None
        )

    # --- 2) Run selection once, regardless of origin ---
    removed_f = select_units(
        context=stand,
        data=sel_data,
        target_decl=target_decl,  # your earlier change already supports None here
        sets=sets_py,
        freq_var="stems_per_ha",
        select_from="all",
        mode="odds_units",
    )

    if np.any(removed_f < 0):
        raise MetsiException("thin_basal_area produced negative removals.")

    removed_f = np.minimum(removed_f, trees.stems_per_ha)
    if not trees.stems_per_ha.flags.writeable:
        trees.stems_per_ha = trees.stems_per_ha.copy()
    trees.stems_per_ha -= removed_f

    metrics_after = compute_stand_metrics(trees)
    removed_G = float(np.nansum(removed_f * sel_data.g))

    cdata.store("thin_basal_area", {
        "time": sim_time,
        "labels": (labels or []) + ["thinning", "cutting"],
        "target_relative_g": float(log_rel_target) if (log_rel_target is not None and np.isfinite(log_rel_target)) else None,
        "removed_stems_per_ha": float(np.nansum(removed_f)),
        "removed_basal_area": removed_G,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "G_instruction_upper": basal_area_instructions_upper_limit(stand),
        "G_instruction_lower": basal_area_instruction_lower_limit(stand),
    })
    return (stand, cdata)