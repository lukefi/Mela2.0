from __future__ import annotations
from typing import Any, Optional
import numpy as np
import numpy.typing as npt

from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.domain.metrics.stand_metrics import compute_stand_metrics
from lukefi.metsi.domain.selection.selection_data import SelectionData
from lukefi.metsi.sim.select_units import SelectionSet, SelectionTarget, select_units
from lukefi.metsi.domain.limit_constants import (
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

def ftrt_thin_basal_area(
    op: OpTuple[ForestStand],
    *,
    # Either give ba_after (target G after thinning) or max_proportion (cap)
    ba_after: Optional[float] = None,
    max_proportion: Optional[float] = None,
    tree_selection: Optional[dict[str, Any]] = None,
    labels: Optional[list[str]] = None,
    sim_time: Optional[int] = None,
) -> OpTuple[ForestStand]:
    """
    Basal-area thinning (generic core).
    - If ba_after is given, global target := max( (G - ba_after)/G, 0 ), optionally capped by max_proportion.
    - If only max_proportion is given, global target := max_proportion (relative on 'g').
    - If neither given, derive ba_after from instruction lower limit (TEMP constants shim).
    - 'tree_selection' may override sets; if None, we default to diameter-ordered from-below thinning.
    """
    stand, cdata = op
    # Get vectors (works with either reference_trees or reference_trees_soa)
    from lukefi.metsi.domain.util.get_reference_trees import get_reference_trees

    trees: Optional[ReferenceTrees] = getattr(stand, "reference_trees", None)
    if trees is None or not isinstance(trees, ReferenceTrees):
        raise MetsiException("thin_basal_area requires vectorized trees in 'reference_trees'.")

    metrics_before = compute_stand_metrics(trees)
    G = metrics_before["G"]  # current basal area m2/ha

    sel_data = SelectionData(trees)

    # --- Compute global target on 'g' (relative) ---
    if ba_after is None and tree_selection is None:
        # derive from instruction lower limit (TEMP)
        ba_after = basal_area_instruction_lower_limit(stand)  # will be table-driven later
    rel_target = np.inf
    if ba_after is not None and G > 0:
        rel_target = max(0.0, min(1.0, (G - ba_after) / G))
    if max_proportion is not None:
        rel_target = min(rel_target, max_proportion) if np.isfinite(rel_target) else max_proportion
    if not np.isfinite(rel_target):
        raise MetsiException("thin_basal_area: unable to determine a global relative target on 'g'.")

    # --- Build defaults if no explicit selection provided ---
    if tree_selection is None:
        # From-below, diameter-ordered; single set that can remove up to 'rel_target' of 'g'
        target_decl = SelectionTarget()
        target_decl.type = "relative"
        target_decl.var = "g"
        target_decl.amount = float(rel_target)

        ss = SelectionSet[ForestStand, SelectionData]()
        ss.sfunction = lambda ctx, data: np.ones(data.size, dtype=bool)
        ss.order_var = "breast_height_diameter"
        ss.target_var = "g"
        ss.target_type = "relative"
        ss.target_amount = 1.0  # consume target across all rows as needed
        ss.profile_x = np.array([0.0, 1.0], dtype=np.float64)
        ss.profile_y = np.array([1.0, 0.0], dtype=np.float64)  # from-below tilt
        ss.profile_xmode = "relative"
        ss.profile_xscale = None
        sets_py = [ss]
    else:
        target = tree_selection["Target"]
        target_decl = SelectionTarget()
        target_decl.type = target["type"]
        target_decl.var = {"g": "g", "f": "stems_per_ha"}.get(target["var"], target["var"])
        target_decl.amount = target["amount"]
        sets_py: list[SelectionSet[ForestStand, SelectionData]] = []
        for s in tree_selection["sets"]:
            ss = SelectionSet[ForestStand, SelectionData]()
            ss.sfunction = s["sfunction"]
            ss.order_var = {"d": "breast_height_diameter", "v": "v"}.get(s["order_var"], s["order_var"])
            ss.target_var = {"g": "g", "f": "stems_per_ha"}.get(s["target_var"], s["target_var"])
            ss.target_type = s["target_type"]
            ss.target_amount = s["target_amount"]
            ss.profile_x = np.asarray(s["profile_x"], dtype=np.float64)
            ss.profile_y = np.asarray(s["profile_y"], dtype=np.float64)
            ss.profile_xmode = s["profile_xmode"]
            ss.profile_xscale = s.get("profile_xscale")
            sets_py.append(ss)

    # --- Select per-row removals (in stems/ha) ---
    removed_f = select_units(
        context=stand,
        data=sel_data,             # exposes 'g' and 'v'
        target_decl=target_decl,
        sets=sets_py,
        freq_var="stems_per_ha",
        select_from="all",
        mode="odds_trees",
    )
    if np.any(removed_f < 0):
        raise MetsiException("thin_basal_area produced negative removals.")

    removed_f = np.minimum(removed_f, trees.stems_per_ha)
    if not trees.stems_per_ha.flags.writeable:
        trees.stems_per_ha = trees.stems_per_ha.copy()
    trees.stems_per_ha -= removed_f

    metrics_after = compute_stand_metrics(trees)
    removed_G = float(np.nansum(removed_f * sel_data.g))  # m2/ha removed
    cdata.store("thin_basal_area", {
        "time": sim_time,
        "labels": (labels or []) + ["thinning", "cutting"],
        "target_relative_g": float(rel_target),
        "removed_stems_per_ha": float(np.nansum(removed_f)),
        "removed_basal_area": removed_G,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        # TEMP: for later event-level checks against instruction limits:
        "G_instruction_upper": basal_area_instructions_upper_limit(stand),
        "G_instruction_lower": basal_area_instruction_lower_limit(stand),
    })
    return (stand, cdata)
