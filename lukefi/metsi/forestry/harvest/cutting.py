from typing import Any

import numpy as np
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.enums.internal import CuttingMethod
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.sim.collected_data import OpTuple, CollectedData
from lukefi.metsi.data.util.select_units import select_units, SelectionSet, SelectionTarget
from lukefi.metsi.domain.collected_data import RemovedTrees
from lukefi.metsi.sim.treatment import Treatment
from lukefi.metsi.domain.natural_processes.grow_motti import (
    apply_motti_yp_reduction_from_removed_reference_trees,
)


def cutting_fn(stand: ForestStand,
               /,
               tree_selection: dict[str, Any] | None = None,
               cutting_method: CuttingMethod | None = None,
               mode: str = "odds_units",
               select_from_all: bool = False,
               ) -> OpTuple[ForestStand]:
    """
    cutting treatment:
      - Updates required stand.cutting_year and stand.method_of_last_cutting.
      - Applies removals to stand.reference_trees.stems_per_ha.
    """

    trees: ReferenceTrees = stand.reference_trees

    if stand.reference_trees.size == 0:
        return stand, []

    if tree_selection is None or "target" not in tree_selection or "sets" not in tree_selection:
        raise MetsiException("Missing 'tree_selection' with 'target' and 'sets'.")

    if cutting_method is None:
        raise MetsiException("Required parameter 'cutting_method' is missing!")

    target: SelectionTarget = tree_selection["target"]

    if stand.year is None:
        raise MetsiException("Stand.year is None!")

    # Sets
    sets: list[SelectionSet[ForestStand, ReferenceTrees]] = tree_selection["sets"]
    if len(sets) == 0:
        raise MetsiException("tree_selection.sets must be a non-empty list.")

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

    # If Motti is in use, update removed trees to yp vector
    if stand.motti_state is not None:
        apply_motti_yp_reduction_from_removed_reference_trees(stand, removed_f)

    stand.cutting_year = stand.year
    stand.method_of_last_cutting = cutting_method

    return stand, collected


cutting = Treatment(cutting_fn, "cutting", collected_data={RemovedTrees})
