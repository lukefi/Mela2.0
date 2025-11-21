import numpy as np
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.sim.collected_data import OpTuple, CollectedData
from lukefi.metsi.data.util.select_units import select_units, SelectionSet, SelectionTarget
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


    target: SelectionTarget = ts["target"]

    if stand.year is None:
        raise MetsiException("Stand.year is None!")

    # Sets
    sets: list[SelectionSet[ForestStand, ReferenceTrees]] = ts["sets"]
    if len(sets) == 0:
        raise MetsiException("tree_selection.sets must be a non-empty list.")

    mode = operation_parameters.get("mode", "odds_units")
    select_from_all = operation_parameters.get("select_from_all", False)

    # Run selection
    removed_f = select_units(
        context=stand,
        data=trees,
        target_decl=target,
        sets=sets,
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
        rt = RemovedTrees()
        rt.removed_trees = removed_view
        collected = [rt]

    return stand, collected
