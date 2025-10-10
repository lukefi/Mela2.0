from __future__ import annotations
from typing import Any, Optional
import numpy as np
import numpy.typing as npt
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.domain.forestry_operations.metrics.stand_metrics import compute_stand_metrics
from lukefi.metsi.data.util.select_units import SelectionSet, SelectionTarget, select_units
from lukefi.metsi.domain.forestry_operations.metrics.selection_data import SelectionData

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

def ftrt_cutting(
    op: OpTuple[ForestStand],
    *,
    tree_selection: dict[str, Any] | None,
    labels: Optional[list[str]] = None,
    sim_time: Optional[int] = None,
) -> OpTuple[ForestStand]:
    """Port of R ftrt_cutting: apply a caller-defined selection; no defaults."""
    stand, cdata = op
    trees: Optional[ReferenceTrees] = getattr(stand, "reference_trees", None)
    if trees is None or not isinstance(trees, ReferenceTrees):
        raise MetsiException("cutting requires vectorized trees in 'reference_trees'.")

    if not tree_selection or "Target" not in tree_selection:
        # R: prints warning and returns node unchanged – we mimic by no-op.
        return (stand, cdata)

    metrics_before = compute_stand_metrics(trees)
    sel_data = SelectionData(trees)

    target = tree_selection["Target"]
    target_decl = SelectionTarget()
    # Map R vars: f->stems_per_ha, g->g, d->breast_height_diameter, v->v
    target_decl.type = target["type"]
    target_decl.var = {"f": "stems_per_ha", "g": "g"}.get(target["var"], target["var"])
    target_decl.amount = target["amount"]

    sets_py: list[SelectionSet[ForestStand, SelectionData]] = []
    for s in tree_selection["sets"]:
        ss = SelectionSet[ForestStand, SelectionData]()
        ss.sfunction = s["sfunction"]
        ss.order_var = {"d": "breast_height_diameter", "v": "v"}.get(s["order_var"], s["order_var"])
        ss.target_var = {"f": "stems_per_ha", "g": "g"}.get(s["target_var"], s["target_var"])
        ss.target_type = s["target_type"]
        ss.target_amount = s["target_amount"]
        ss.profile_x = np.asarray(s["profile_x"], dtype=np.float64)
        ss.profile_y = np.asarray(s["profile_y"], dtype=np.float64)
        ss.profile_xmode = s["profile_xmode"]
        ss.profile_xscale = s.get("profile_xscale")
        sets_py.append(ss)

    removed_f = select_units(
        context=stand,
        data=sel_data,
        target_decl=target_decl,
        sets=sets_py,
        freq_var="stems_per_ha",
        select_from="all",
        mode="odds_trees",
    )
    if np.any(removed_f < 0):
        raise MetsiException("cutting produced negative removals.")

    removed_f = np.minimum(removed_f, trees.stems_per_ha)
    if not trees.stems_per_ha.flags.writeable:
        trees.stems_per_ha = trees.stems_per_ha.copy()
    trees.stems_per_ha -= removed_f

    metrics_after = compute_stand_metrics(trees)
    cdata.extend_list_result("removed_trees", [_removed_snapshot(trees, removed_f)])
    cdata.store("cutting", {
        "time": sim_time,
        "labels": (labels or []) + ["cutting"],
        "removed_stems_per_ha": float(np.nansum(removed_f)),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    })
    return (stand, cdata)
z