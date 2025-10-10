from __future__ import annotations
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.sim.collected_data import CollectedData, OpTuple
from lukefi.metsi.domain.metrics.stand_metrics import compute_stand_metrics
from lukefi.metsi.sim.select_units import SelectionSet, select_units  # adjust import to your layout

def _removed_snapshot(trees: ReferenceTrees, removed_f: npt.NDArray[np.float64]) -> dict[str, npt.NDArray]:
    """Return a compact snapshot of removed rows for reporting."""
    mask = removed_f > 0
    if not np.any(mask):
        return {"stems_per_ha": np.zeros(0, dtype=np.float64)}
    return {
        "stems_per_ha": removed_f[mask],
        "breast_height_diameter": trees.breast_height_diameter[mask],
        "height": getattr(trees, "height", None)[mask] if hasattr(trees, "height") else None,
        "species": getattr(trees, "species", None)[mask] if hasattr(trees, "species") else None,
    }

def ftrt_ajourat(
    op: OpTuple[ForestStand],
    *,
    proportion: float = 0.18,
    tree_selection: Optional[dict[str, Any]] = None,
    labels: Optional[list[str]] = None,
    sim_time: Optional[int] = None,
) -> OpTuple[ForestStand]:
    """
    Even removal for strip-road establishment (ajourat).
    - Removes a relative share of stems across diameter profile (default 18%).
    - Uses select_units to produce per-row removals in stems/ha.
    - Mutates only the 'stems_per_ha' vector; everything else is untouched.
    - Records before/after stand metrics and a compact 'removed' snapshot in CollectedData.
    """
    stand, cdata = op

    # Expect vector trees to be present on the stand
    trees: Optional[ReferenceTrees] = getattr(stand, "reference_trees", None)
    if trees is None or not isinstance(trees, ReferenceTrees):
        raise MetsiException("ajourat requires vectorized trees in 'reference_trees'.")

    # Metrics BEFORE
    metrics_before = compute_stand_metrics(trees)

    # Build default selection if one wasn’t provided (flat profile over diameter, relative target on stems/ha)
    if tree_selection is None:
        tree_selection = {
            "Target": {"type": None, "var": None, "amount": None},
            "sets": [
                {
                    "sfunction": lambda ctx, data: np.ones(data.size, dtype=bool),
                    "order_var": "breast_height_diameter",
                    "target_var": "stems_per_ha",
                    "target_type": "relative",
                    "target_amount": float(proportion),
                    "profile_x": np.array([0.0, 1.0]),
                    "profile_y": np.array([0.5, 0.5]),
                    "profile_xmode": "relative",
                }
            ],
        }

    # Adapt SelectionSet list for select_units
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

    # Run selection; receive per-row removals (stems/ha)
    removed_f = select_units(
        context=stand,
        data=trees,
        target_decl=None,              # no global target for ajourat
        sets=sets_py,
        freq_var="stems_per_ha",
        select_from="all",
        mode="odds_trees",
    )

    if np.any(removed_f < 0):
        raise MetsiException("ajourat produced negative removals; check selection profile/targets.")

    # Apply (clamp to available stems to be safe)
    removed_f = np.minimum(removed_f, trees.stems_per_ha)
    if not trees.stems_per_ha.flags.writeable:
        trees.stems_per_ha = trees.stems_per_ha.copy()
    trees.stems_per_ha -= removed_f

    # Metrics AFTER
    metrics_after = compute_stand_metrics(trees)

    # Collect reporting
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
