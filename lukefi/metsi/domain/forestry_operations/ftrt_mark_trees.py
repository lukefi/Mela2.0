from __future__ import annotations
from typing import Any, Optional
import numpy as np
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.data.utils.select_units import SelectionSet, SelectionTarget, select_units
from lukefi.metsi.domain.forestry_operationsselection.selection_data import SelectionData

def ftrt_mark_trees(
    op: OpTuple[ForestStand],
    *,
    tree_selection: dict[str, Any],
    attributes: dict[str, Any],     # e.g., {"tree_type": "retained", "management_category": 2}
    labels: Optional[list[str]] = None,
    sim_time: Optional[int] = None,
) -> OpTuple[ForestStand]:
    """Mark selected stems by cloning rows with new attributes and reducing the originals accordingly (total stems conserved)."""
    stand, cdata = op
    trees: Optional[ReferenceTrees] = getattr(stand, "reference_trees", None)
    if trees is None or not isinstance(trees, ReferenceTrees):
        raise MetsiException("mark_trees requires vectorized trees in 'reference_trees'.")
    if not tree_selection or "Target" not in tree_selection:
        return (stand, cdata)

    sel_data = SelectionData(trees)

    # Build selection
    target = tree_selection["Target"]
    target_decl = SelectionTarget()
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

    to_mark = select_units(
        context=stand,
        data=sel_data,
        target_decl=target_decl,
        sets=sets_py,
        freq_var="stems_per_ha",
        select_from="all",
        mode="odds_trees",
    )
    to_mark = np.minimum(to_mark, trees.stems_per_ha)
    if not np.any(to_mark > 0):
        return (stand, cdata)

    # Subtract from originals
    if not trees.stems_per_ha.flags.writeable:
        trees.stems_per_ha = trees.stems_per_ha.copy()
    trees.stems_per_ha -= to_mark

    # Clone selected rows with new attributes
    mask = to_mark > 0
    idxs = np.nonzero(mask)[0]
    new_rows = []
    for i in idxs:
        row = {k: trees[k][i] for k in trees.dtypes.keys()}
        row["stems_per_ha"] = float(to_mark[i])
        # apply attribute overrides
        for k, v in attributes.items():
            if k in trees.dtypes:
                row[k] = v
        new_rows.append(row)
    trees.create(new_rows)

    cdata.store("mark_trees", {
        "time": sim_time,
        "labels": labels or [],
        "marked_stems_per_ha": float(np.nansum(to_mark)),
        "attributes": attributes,
    })
    return (stand, cdata)
