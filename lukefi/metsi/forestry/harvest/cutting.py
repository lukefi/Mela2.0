import numpy as np
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.sim.collected_data import OpTuple, CollectedData
from lukefi.metsi.data.util.select_units import select_units, SelectionSet, SelectionTarget
from lukefi.metsi.forestry.treatment_utils import req
from lukefi.metsi.domain.collected_data import RemovedTrees

def cutting(input_: ForestStand, /, **operation_parameters) -> OpTuple[ForestStand]:
    """
    cutting treatment:
      - Requires operation_parameters['tree_selection'] with:
          Target: {type, var, amount}
          sets: [ {sfunction, order_var, target_var, target_type, target_amount,
                   profile_x, profile_y, profile_xmode, (optional) profile_xscale}, ... ]
      - Updates required stand.cutting_year and stand.method_of_last_cutting.
      - Applies removals to stand.reference_trees.stems_per_ha.
    """
    stand = input_

    trees: ReferenceTrees = stand.reference_trees
    if not isinstance(trees, ReferenceTrees):
        raise MetsiException("cutting requires stand.reference_trees.")

    if stand.reference_trees.size == 0:
        return stand, []


    ts = operation_parameters.get("tree_selection")
    if not ts or "target" not in ts or "sets" not in ts:
        raise MetsiException("Missing 'tree_selection' with 'target' and 'sets'.")

    method = operation_parameters.get("cutting_method")
    if method is None:
        raise MetsiException("Required parameter 'cutting_method' is missing!")


    target = ts["target"]
    for k in ("type", "var", "amount"):
        if k not in target:
            raise MetsiException(f"tree_selection.Target missing '{k}'.")

    if stand.year is None:
        raise MetsiException("Stand.year is None!")

    # Global target
    target_decl = SelectionTarget()
    target_decl.type = target["type"]
    target_decl.var = target["var"]
    target_decl.amount = target["amount"]

    # Sets
    sets_in = ts["sets"]
    if not isinstance(sets_in, (list, tuple)) or len(sets_in) == 0:
        raise MetsiException("tree_selection.sets must be a non-empty list.")

    required = ("sfunction", "order_var", "target_var", "target_type",
                "target_amount", "profile_x", "profile_y", "profile_xmode")

    py_sets: list[SelectionSet[ForestStand, ReferenceTrees]] = []
    for i, s in enumerate(sets_in):
        # Checking if item exists
        for name in required:
            req(s, name)
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
        if ss.profile_x.shape != ss.profile_y.shape or ss.profile_x.ndim != 1 or ss.profile_x.size < 2:
            raise MetsiException(f"sets[{i}]: profile_x/profile_y must be 1D arrays of equal length (>=2).")
        py_sets.append(ss)

    mode = operation_parameters.get("mode", "odds_units")
    select_from_all = operation_parameters.get("select_from_all", False)

    # Run selection
    removed_f = select_units(
        context=stand,
        data=trees,
        target_decl=target_decl,
        sets=py_sets,
        freq_var="stems_per_ha",
        select_from_all=select_from_all,
        mode=mode,
    )

    # Apply removals
    if not trees.stems_per_ha.flags.writeable:
        trees.stems_per_ha = trees.stems_per_ha.copy()
    trees.stems_per_ha -= removed_f

    stand.cutting_year = stand.year
    stand.method_of_last_cutting = method

    # Collected data: Removed trees
    removed_mask = removed_f > 0.0
    collected: list[CollectedData] = []
    if np.any(removed_mask):
        removed_view = trees[removed_mask]
        # record the removed amounts as stems_per_ha in the collected view
        removed_view.stems_per_ha = removed_f[removed_mask].copy()
        rt: CollectedData = RemovedTrees()
        rt.removed_trees = removed_view
        collected = [rt]

    return stand, collected
