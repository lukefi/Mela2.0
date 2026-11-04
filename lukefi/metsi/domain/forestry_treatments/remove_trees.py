from typing import Any
import numpy as np

from lukefi.metsi.core.treatment import Treatment
from lukefi.metsi.core.utils.select_units import Mode, SelectionSet, SelectionTarget, select_units
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.core.collected_data import CollectedData, OpTuple
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.domain.collected_data import RemovedTrees
from lukefi.metsi.domain.natural_processes.motti_util import apply_motti_yp_reduction_from_removed_reference_trees
from lukefi.metsi.forestry.aggregates_internal import calc_tree_basal_areas
from lukefi.metsi.forestry.storey import manage_existing_storeys


def remove_trees_fn(stand: ForestStand,
                 *,
                 tree_selection_sets: list[SelectionSet[ForestStand, ReferenceTrees]] | None = None,
                 tree_selection_target: SelectionTarget | None = None,
                 tree_selection_mode: Mode = Mode.ODDS_UNITS,
                 select_from_all: bool = False,
                 updated_attributes: dict[str, Any] | None = None) -> OpTuple[ForestStand]:

    assert tree_selection_sets is not None
    assert tree_selection_target is not None
    assert len(tree_selection_sets) > 0

    if updated_attributes is None:
        updated_attributes = {}

    trees = stand.reference_trees

    removed_stems = select_units(context=stand,
                                 data=trees,
                                 target_decl=tree_selection_target,
                                 sets=tree_selection_sets,
                                 freq_var="stems_per_ha",
                                 select_from_all=select_from_all,
                                 mode=tree_selection_mode)

    # Apply removals
    if not trees.stems_per_ha.flags.writeable:
        trees.stems_per_ha = trees.stems_per_ha.copy()
        trees.stems_per_ha.flags.writeable = True
    trees.stems_per_ha -= removed_stems

    # Collected data
    removed_mask = removed_stems > 0
    collected: list[CollectedData] = []

    if np.any(removed_mask):
        removed_view = trees[removed_mask]
        removed_view.stems_per_ha = removed_stems[removed_mask].copy()
        rt = RemovedTrees()
        rt.removed_trees = removed_view
        collected.append(rt)

    if stand.motti_state is not None:
        apply_motti_yp_reduction_from_removed_reference_trees(stand, removed_stems)

    for attribute, new_value in updated_attributes.items():
        setattr(stand, attribute, new_value)

    trees.basal_area = calc_tree_basal_areas(trees.breast_height_diameter)
    manage_existing_storeys(trees)

    return stand, collected


remove_trees = Treatment(remove_trees_fn, "remove_trees", collected_data={RemovedTrees})
