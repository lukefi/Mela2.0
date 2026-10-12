from functools import wraps
from typing import Callable, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from lukefi.metsi.core.collected_data import CollectedData
from lukefi.metsi.core.transition import TransitionFn
from lukefi.metsi.data.enums.internal import (
    Storey,
    TreeManagementCategory,
    TreeSpecies)
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.forestry.aggregates import (
    calc_basal_area,
    calc_mean_height,
    calc_tree_basal_areas,
    determine_dominant_species,
    should_use_ba_as_weight)

if TYPE_CHECKING:
    from lukefi.metsi.data.model import ForestStand

DOMINANT_STOREY_BA_LIMIT = 1.0  # m^2/ha
DOMINANT_STOREY_STEMS_LIMIT = 400.0  # 1/ha


def stand_has_only_retention_storey(trees: ReferenceTrees) -> bool:
    return bool(np.all(trees.storey == Storey.RETENTION))


def stand_has_only_seeding_tree_storey(trees: ReferenceTrees) -> bool:
    return bool(np.all(
        (trees.storey == Storey.OVER) & (trees.management_category == TreeManagementCategory.SEEDING_TREE)))


def promote_either_storey_to_dominant(trees: ReferenceTrees,
                                      mask1: npt.NDArray[np.bool_],
                                      mask2: npt.NDArray[np.bool_],
                                      fallback: Callable[[ReferenceTrees, bool], None]):

    storey_1_tree_diameters = trees.breast_height_diameter[mask1]
    storey_1_tree_basal_areas = trees.basal_area[mask1]
    storey_1_tree_species = trees.species[mask1]
    storey_1_tree_stems_per_ha = trees.stems_per_ha[mask1]

    storey_2_tree_diameters = trees.breast_height_diameter[mask2]
    storey_2_tree_basal_areas = trees.basal_area[mask2]
    storey_2_tree_species = trees.species[mask2]
    storey_2_tree_stems_per_ha = trees.stems_per_ha[mask2]

    compare_storey_1_by_ba = should_use_ba_as_weight(
        storey_1_tree_diameters,
        storey_1_tree_basal_areas,
        storey_1_tree_stems_per_ha)
    compare_storey_2_by_ba = should_use_ba_as_weight(
        storey_2_tree_diameters,
        storey_2_tree_basal_areas,
        storey_2_tree_stems_per_ha
    )

    if compare_storey_1_by_ba != compare_storey_2_by_ba:
        # Can not compare storeys, use fallback
        fallback(trees, compare_storey_1_by_ba)
        return

    storey_1_dominant_species = determine_dominant_species(storey_1_tree_diameters,
                                                           storey_1_tree_stems_per_ha,
                                                           storey_1_tree_species,
                                                           storey_1_tree_basal_areas)
    storey_2_dominant_species = determine_dominant_species(storey_2_tree_diameters,
                                                           storey_2_tree_stems_per_ha,
                                                           storey_2_tree_species,
                                                           storey_2_tree_basal_areas)

    if storey_1_dominant_species == TreeSpecies.TREELESS and storey_2_dominant_species == TreeSpecies.TREELESS:
        fallback(trees, compare_storey_1_by_ba)
        return
    if storey_1_dominant_species == TreeSpecies.TREELESS:
        trees.storey[mask2] = Storey.DOMINANT
        return
    if storey_2_dominant_species == TreeSpecies.TREELESS:
        trees.storey[mask1] = Storey.DOMINANT
        return

    storey_1_dominant_species_mask = storey_1_tree_species == storey_1_dominant_species
    storey_2_dominant_species_mask = storey_2_tree_species == storey_2_dominant_species

    if compare_storey_1_by_ba:  # and compare_storey_2_by_ba
        # Compare by basal area

        storey_1_dominant_species_ba = calc_basal_area(
            storey_1_tree_basal_areas[storey_1_dominant_species_mask],
            storey_1_tree_stems_per_ha[storey_1_dominant_species_mask])
        storey_2_dominant_species_ba = calc_basal_area(
            storey_2_tree_basal_areas[storey_2_dominant_species_mask],
            storey_2_tree_stems_per_ha[storey_2_dominant_species_mask])

        if storey_1_dominant_species_ba >= storey_2_dominant_species_ba:
            # Promote storey 1
            trees.storey[mask1] = Storey.DOMINANT
            return

        # Promote storey 2
        trees.storey[mask2] = Storey.DOMINANT
        return

    # Compare by stems
    storey_1_dominant_species_stems = np.sum(storey_1_tree_stems_per_ha[storey_1_dominant_species_mask])
    storey_2_dominant_species_stems = np.sum(storey_2_tree_stems_per_ha[storey_2_dominant_species_mask])

    if storey_1_dominant_species_stems >= storey_2_dominant_species_stems:
        # Promote storey 1
        trees.storey[mask1] = Storey.DOMINANT
        return

    # Promote storey 2
    trees.storey[mask2] = Storey.DOMINANT


def manage_existing_storeys(trees: ReferenceTrees):

    if trees.storey.flags.writeable is False:
        trees.storey = np.copy(trees.storey)
        trees.storey.flags.writeable = True

    # Check diminished DOMINANT storey ------------------------------------------------------------------

    storeys = np.unique(trees.storey)

    if _dominant_storey_is_too_small(trees):

        # DOMINANT storey is too small
        if Storey.UNDER in storeys:

            # Make UNDER storey DOMINANT
            trees.storey[trees.storey == Storey.UNDER] = Storey.DOMINANT

            # Recheck if should also merge OVER storey
            if (Storey.OVER in storeys and
                np.all(trees.management_category[trees.storey == Storey.OVER] !=
                       TreeManagementCategory.SEEDING_TREE)):

                if _dominant_storey_is_too_small(trees):
                    # Merge OVER storey to DOMINANT
                    trees.storey[trees.storey == Storey.OVER] = Storey.DOMINANT

        elif (Storey.OVER in storeys and
                np.all(trees.management_category[trees.storey == Storey.OVER] !=
                       TreeManagementCategory.SEEDING_TREE)):

            # Make OVER storey DOMINANT
            trees.storey[trees.storey == Storey.OVER] = Storey.DOMINANT

    # Merge existing storeys ----------------------------------------------------------------------------

    storeys = np.unique(trees.storey[np.isin(trees.storey, (Storey.DOMINANT, Storey.UNDER, Storey.OVER))])
    mean_heights = {storey: calc_mean_height(trees, trees.storey == storey) for storey in storeys}

    dominant_storey_mean_height = mean_heights.get(Storey.DOMINANT, 0.0)

    if Storey.UNDER in storeys:
        if abs(dominant_storey_mean_height - mean_heights[Storey.UNDER]) < 5.0:
            # Merge UNDER into DOMINANT
            trees.storey[trees.storey == Storey.UNDER] = Storey.DOMINANT

    if Storey.OVER in storeys:
        if abs(mean_heights[Storey.OVER] - dominant_storey_mean_height) < 5.0:
            # Merge OVER into DOMINANT
            trees.storey[trees.storey == Storey.OVER] = Storey.DOMINANT


def update_storeys(trees: ReferenceTrees):

    if trees.storey.flags.writeable is False:
        trees.storey = np.copy(trees.storey)
        trees.storey.flags.writeable = True

    if np.any(trees.storey == Storey.UNSET):
        storeys = np.unique(trees.storey)
        has_dominant_storey = Storey.DOMINANT in storeys

        # Merge new trees into DOMINANT or UNDER storey
        if not has_dominant_storey:
            # Mark new trees as DOMINANT
            trees.storey[trees.storey == Storey.UNSET] = Storey.DOMINANT
        else:
            # Check if should merge new trees into DOMINANT
            diff = calc_mean_height(trees, trees.storey == Storey.DOMINANT) - \
                calc_mean_height(trees, trees.storey == Storey.UNSET)
            if abs(diff) < 5.0:
                # Merge into DOMINANT
                trees.storey[trees.storey == Storey.UNSET] = Storey.DOMINANT
            else:
                # Merge into UNDER
                trees.storey[trees.storey == Storey.UNSET] = Storey.UNDER

    manage_existing_storeys(trees)


def handle_storeys_after_natural_process(natural_process_func: TransitionFn["ForestStand"]):

    @wraps(natural_process_func)
    def wrapper(unit: "ForestStand", step: int, **params) -> tuple["ForestStand", list[CollectedData]]:
        unit, cd = natural_process_func(unit, step, **params)
        trees = unit.reference_trees

        if len(trees) == 0:
            return unit, cd

        if not trees.storey.flags.writeable:
            trees.storey = np.copy(trees.storey)
            trees.storey.flags.writeable = True

        # Pre-calculate basal area for all trees
        trees.basal_area = calc_tree_basal_areas(trees.breast_height_diameter)

        update_storeys(trees)

        return unit, cd

    return wrapper


def _dominant_storey_is_too_small(trees: ReferenceTrees) -> bool:
    if np.all(trees.storey != Storey.DOMINANT):
        # No DOMINANT storey at all
        return True

    dominant_storey_mask = trees.storey == Storey.DOMINANT
    dominant_storey_diameters = trees.breast_height_diameter[dominant_storey_mask]
    dominant_storey_basal_areas = trees.basal_area[dominant_storey_mask]
    dominant_storey_stems_per_ha = trees.stems_per_ha[dominant_storey_mask]

    should_use_ba = should_use_ba_as_weight(
        dominant_storey_diameters,
        dominant_storey_basal_areas,
        dominant_storey_stems_per_ha
    )

    if should_use_ba:
        storey_basal_area = calc_basal_area(dominant_storey_basal_areas, dominant_storey_stems_per_ha)
        if storey_basal_area < DOMINANT_STOREY_BA_LIMIT:
            return True
        return False

    storey_stems_per_ha = np.sum(dominant_storey_stems_per_ha)
    if storey_stems_per_ha < DOMINANT_STOREY_STEMS_LIMIT:
        return True

    return False
