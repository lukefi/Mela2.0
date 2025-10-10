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

def ftrt_thin_nr_of_stems(
    op: OpTuple[ForestStand],
    *,
    max_proportion: float,
    stems_after: float,
    tree_selection: Optional[dict[str, Any]] = None,
    labels: Optional[list[str]] = None,
    sim_time: Optional[int] = None,
) -> OpTuple[ForestStand]:
    """
    Port of R ftrt_thin_nr_of_stems: prioritize (1) d>15 & manag_cat<=1, (2) non-dominant species & manag_cat<=1,
    (3) manag_cat<=1, targeting remaining stems >= stems_after and <= max_proportion removed.
    """
    stand, cdata = op
    trees: Optional[ReferenceTrees] = getattr(stand, "reference_trees", None)
    if trees is None or not isinstance(trees, ReferenceTrees):
        raise MetsiException("thin_nr_of_stems requires vectorized trees in 'reference_trees'.")

    metrics_before = compute_stand_metrics(trees)
    dom_spe = metrics_before["dom_spe"]

    sel_data = SelectionData(trees)
    total_f = float(np.nansum(sel_data["stems_per_ha"]))
    target_amount = max(stems_after, (1 - max_proportion) * total_f)

    if tree_selection is None:
        # Build default R selection (three sets, in priority order)
        target_decl = SelectionTarget()
        target_decl.type = "absolute_remain"
        target_decl.var = "stems_per_ha"   # f
        target_decl.amount = target_amount

        def set1(ctx, data):  # d > 15 & manag_cat <= 1
            return (data["breast_height_diameter"] > 15.0) & (data["management_category"] <= 1)

        def set2(ctx, data):  # spe != dom_spe & manag_cat <= 1
            if dom_spe is None:
                return data["management_category"] <= 1
            return (data["species"] != dom_spe) & (data["management_category"] <= 1)

        def set3(ctx, data):  # manag_cat <= 1
            return data["management_category"] <= 1

        def mkset(sfunc, order_var, target_var, prof_y0, prof_y1):
            ss = SelectionSet[ForestStand, SelectionData]()
            ss.sfunction = sfunc
            ss.order_var = order_var
            ss.target_var = target_var
            ss.target_type = "relative"
            ss.target_amount = max_proportion
            ss.profile_x = np.array([0.0, 1.0], dtype=np.float64)
            ss.profile_y = np.array([prof_y0, prof_y1], dtype=np.float64)
            ss.profile_xmode = "relative"
            ss.profile_xscale = None
            return ss

        sets_py = [
            mkset(set1, "breast_height_diameter", "stems_per_ha", 0.01, 0.99),
            mkset(set2, "breast_height_diameter", "stems_per_ha", 1.00, 0.50),
            mkset(set3, "breast_height_diameter", "stems_per_ha", 1.00, 0.50),
        ]
    else:
        # Advanced caller-provided selection (rare)
        target = tree_selection["Target"]
        target_decl = SelectionTarget()
        target_decl.type = target["type"]
        target_decl.var = "stems_per_ha" if target["var"] == "f" else target["var"]
        target_decl.amount = target["amount"]
        sets_py = []
        for s in tree_selection["sets"]:
            ss = SelectionSet[ForestStand, SelectionData]()
            ss.sfunction = s["sfunction"]
            ss.order_var = "breast_height_diameter" if s["order_var"] == "d" else s["order_var"]
            ss.target_var = "stems_per_ha" if s["target_var"] == "f" else s["target_var"]
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
        mode="odds_units",
    )
    if np.any(removed_f < 0):
        raise MetsiException("thin_nr_of_stems produced negative removals.")

    removed_f = np.minimum(removed_f, trees.stems_per_ha)
    if not trees.stems_per_ha.flags.writeable:
        trees.stems_per_ha = trees.stems_per_ha.copy()
    trees.stems_per_ha -= removed_f

    metrics_after = compute_stand_metrics(trees)
    cdata.extend_list_result("removed_trees", [_removed_snapshot(trees, removed_f)])
    cdata.store("thin_nr_of_stems", {
        "time": sim_time,
        "labels": (labels or []) + ["first_thinning", "thinning", "cutting"],
        "max_proportion": max_proportion,
        "stems_after_target": stems_after,
        "removed_stems_per_ha": float(np.nansum(removed_f)),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        # note: dom_spe used for selection
        "dom_spe_before": dom_spe,
    })
    return (stand, cdata)
