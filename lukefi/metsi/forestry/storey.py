from functools import wraps
from typing import Callable, Set

import numpy as np
import numpy.typing as npt

from lukefi.metsi.core.collected_data import CollectedData
from lukefi.metsi.core.transition import TransitionFn
from lukefi.metsi.data.enums.internal import (
    CONIFEROUS_SPECIES,
    DECIDUOUS_SPECIES,
    Storey,
    TreeManagementCategory,
    TreeSpecies)
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees

DOMINANT_STOREY_BA_LIMIT = 1.0  # m^2/ha
DOMINANT_STOREY_STEMS_LIMIT = 400.0  # 1/ha


def stand_has_only_retention_storey(trees: ReferenceTrees) -> bool:
    return bool(np.all(trees.storey == Storey.RETENTION))


def stand_has_only_seeding_tree_storey(trees: ReferenceTrees) -> bool:
    return bool(np.all(
        (trees.storey == Storey.OVER) & (trees.management_category == TreeManagementCategory.SEEDING_TREE)))


def calc_tree_basal_areas(diameters: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.pi * (diameters / 200) ** 2


def calc_storey_basal_area(tree_basal_areas: npt.NDArray[np.float64], stems_per_ha: npt.NDArray[np.float64]) -> float:
    return np.sum(tree_basal_areas * stems_per_ha)


def should_use_ba_for_storey(diameters: npt.NDArray[np.float64],
                             basal_areas: npt.NDArray[np.float64],
                             stems_per_ha: npt.NDArray[np.float64]) -> bool:
    if np.sum(basal_areas) == 0:
        return False

    storey_mean_diameter = np.sum(diameters * basal_areas * stems_per_ha) / np.sum(basal_areas * stems_per_ha)
    return bool(storey_mean_diameter >= 8.0)


def determine_dominant_species(tree_diameters: npt.NDArray[np.float64],
                               tree_stems_per_ha: npt.NDArray[np.float64],
                               tree_species: npt.NDArray[np.int32],
                               tree_basal_areas: npt.NDArray[np.float64]) -> TreeSpecies:

    if should_use_ba_for_storey(tree_diameters, tree_basal_areas, tree_stems_per_ha):
        # Use basal area for comparing
        storey_basal_area = calc_storey_basal_area(tree_basal_areas, tree_stems_per_ha)
        if storey_basal_area < DOMINANT_STOREY_BA_LIMIT:
            return TreeSpecies.TREELESS

        coniferous_mask = np.isin(tree_species, CONIFEROUS_SPECIES)
        coniferous_diameters = tree_diameters[coniferous_mask]
        coniferous_stems_per_ha = tree_stems_per_ha[coniferous_mask]
        coniferous_ba = np.pi * np.sum(((coniferous_diameters / 200) ** 2) * coniferous_stems_per_ha)

        if coniferous_ba >= storey_basal_area / 2:
            # Coniferous dominated storey
            species_list = CONIFEROUS_SPECIES
        else:
            # Deciduous dominated storey
            species_list = DECIDUOUS_SPECIES

        species_bas = {species: calc_storey_basal_area(
            tree_basal_areas[tree_species == species], tree_stems_per_ha[tree_species == species])
            for species in species_list}

        # Species with maximum basal area
        return max(species_bas, key=lambda key: species_bas[key])

    # Use stems per ha for comparing
    storey_stems_per_ha = np.sum(tree_stems_per_ha)
    if storey_stems_per_ha < DOMINANT_STOREY_STEMS_LIMIT:
        return TreeSpecies.TREELESS

    coniferous_mask = np.isin(tree_species, CONIFEROUS_SPECIES)
    coniferous_stems_per_ha = tree_stems_per_ha[coniferous_mask]

    if coniferous_stems_per_ha >= storey_stems_per_ha / 2:
        # Coniferous dominated storey
        species_list = CONIFEROUS_SPECIES
    else:
        # Deciduous dominated storey
        species_list = DECIDUOUS_SPECIES

    species_stems = {species: np.sum(tree_stems_per_ha[tree_species == species]) for species in species_list}

    # Species with maximum stems per ha
    return max(species_stems, key=lambda key: species_stems[key])


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

    compare_storey_1_by_ba = should_use_ba_for_storey(
        storey_1_tree_diameters,
        storey_1_tree_basal_areas,
        storey_1_tree_stems_per_ha)
    compare_storey_2_by_ba = should_use_ba_for_storey(
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

    storey_1_dominant_species_mask = storey_1_tree_species == storey_1_dominant_species
    storey_2_dominant_species_mask = storey_2_tree_species == storey_2_dominant_species

    if compare_storey_1_by_ba:  # and compare_storey_2_by_ba
        # Compare by basal area

        storey_1_dominant_species_ba = calc_storey_basal_area(
            storey_1_tree_basal_areas[storey_1_dominant_species_mask],
            storey_1_tree_stems_per_ha[storey_1_dominant_species_mask])
        storey_2_dominant_species_ba = calc_storey_basal_area(
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
    storey_1_dominant_species_stems = storey_1_tree_stems_per_ha[storey_1_dominant_species_mask]
    storey_2_dominant_species_stems = storey_2_tree_stems_per_ha[storey_2_dominant_species_mask]

    if storey_1_dominant_species_stems >= storey_2_dominant_species_stems:
        # Promote storey 1
        trees.storey[mask1] = Storey.DOMINANT
        return

    # Promote storey 2
    trees.storey[mask2] = Storey.DOMINANT


def calculate_storey_mean_heights(trees: ReferenceTrees, storeys: Set[Storey]) -> dict[Storey, float]:
    retval: dict[Storey, float] = {}
    for storey in storeys:
        if storey == Storey.RETENTION:
            # Retention storey will not change
            continue
        storey_mask = trees.storey == storey
        tree_diameters = trees.breast_height_diameter[storey_mask]
        tree_basal_areas = trees.basal_area[storey_mask]
        tree_stems_per_ha = trees.stems_per_ha[storey_mask]
        tree_heights = trees.height[storey_mask]

        should_use_ba = should_use_ba_for_storey(tree_diameters, tree_basal_areas, tree_stems_per_ha)
        if should_use_ba:
            mean_height = np.sum(tree_basal_areas * tree_stems_per_ha * tree_heights) / \
                np.sum(tree_basal_areas * tree_stems_per_ha)
        else:
            mean_height = np.sum(tree_heights * tree_stems_per_ha) / np.sum(tree_stems_per_ha)
        retval[storey] = mean_height
    return retval


def handle_storeys_after_natural_process(natural_process_func: TransitionFn[ForestStand]):

    @wraps(natural_process_func)
    def wrapper(unit: ForestStand, step: int, **params) -> tuple[ForestStand, list[CollectedData]]:
        unit, cd = natural_process_func(unit, step, **params)
        trees = unit.reference_trees

        if len(trees) == 0:
            return unit, cd

        if not trees.storey.flags.writeable:
            trees.storey = np.copy(trees.storey)
            trees.storey.flags.writeable = True

        # Pre-calculate basal area for all trees
        trees.basal_area = calc_tree_basal_areas(trees.breast_height_diameter)

        # Handle new trees' storeys --------------------------------------------------------------------------

        if np.any(trees.storey == Storey.UNSET):
            storeys = np.unique(trees.storey)
            has_dominant_storey = Storey.DOMINANT in storeys

            # Merge new trees into DOMINANT or UNDER storey
            if not has_dominant_storey:
                # Mark new trees as DOMINANT
                trees.storey[trees.storey == Storey.UNSET] = Storey.DOMINANT
            else:
                # Check if should merge new trees into DOMINANT
                mean_heights = calculate_storey_mean_heights(trees, {Storey.DOMINANT, Storey.UNSET})
                diff = mean_heights[Storey.DOMINANT] - mean_heights[Storey.UNSET]
                if abs(diff) < 5.0:
                    # Merge into DOMINANT
                    trees.storey[trees.storey == Storey.UNSET] = Storey.DOMINANT
                else:
                    # Merge into UNDER
                    trees.storey[trees.storey == Storey.UNSET] = Storey.UNDER

        # Check diminished DOMINANT storey ------------------------------------------------------------------

        storeys = np.unique(trees.storey)

        if Storey.DOMINANT in storeys:
            dominant_storey_mask = trees.storey == Storey.DOMINANT
            dominant_storey_diameters = trees.breast_height_diameter[dominant_storey_mask]
            dominant_storey_basal_areas = trees.basal_area[dominant_storey_mask]
            dominant_storey_stems_per_ha = trees.stems_per_ha[dominant_storey_mask]

            should_use_ba = should_use_ba_for_storey(
                dominant_storey_diameters,
                dominant_storey_basal_areas,
                dominant_storey_stems_per_ha)

            if ((should_use_ba and
                calc_storey_basal_area(dominant_storey_basal_areas,
                                       dominant_storey_stems_per_ha) < DOMINANT_STOREY_BA_LIMIT) or
                    (not should_use_ba and np.sum(dominant_storey_stems_per_ha) < DOMINANT_STOREY_STEMS_LIMIT)):

                # DOMINANT storey is too small
                if Storey.UNDER in storeys:

                    # Make UNDER storey DOMINANT
                    trees.storey[trees.storey == Storey.UNDER] = Storey.DOMINANT

                    # Recheck if should also merge OVER storey
                    if (Storey.OVER in storeys and
                        np.all(trees.management_category[trees.storey == Storey.OVER] !=
                               TreeManagementCategory.SEEDING_TREE)):

                        dominant_storey_mask = trees.storey == Storey.DOMINANT
                        dominant_storey_diameters = trees.breast_height_diameter[dominant_storey_mask]
                        dominant_storey_basal_areas = trees.basal_area[dominant_storey_mask]
                        dominant_storey_stems_per_ha = trees.basal_area[dominant_storey_mask]

                        should_use_ba = should_use_ba_for_storey(
                            dominant_storey_diameters,
                            dominant_storey_basal_areas,
                            dominant_storey_stems_per_ha
                        )

                        if ((should_use_ba and
                            calc_storey_basal_area(dominant_storey_basal_areas,
                                                   dominant_storey_stems_per_ha) < DOMINANT_STOREY_BA_LIMIT) or
                                (not should_use_ba and
                                    np.sum(dominant_storey_stems_per_ha) < DOMINANT_STOREY_STEMS_LIMIT)):

                            # Merge OVER storey to DOMINANT
                            trees.storey[trees.storey == Storey.OVER] = Storey.DOMINANT

                elif (Storey.OVER in storeys and
                        np.all(trees.management_category[trees.storey == Storey.OVER] !=
                               TreeManagementCategory.SEEDING_TREE)):

                    # Make OVER storey DOMINANT
                    trees.storey[trees.storey == Storey.OVER] = Storey.DOMINANT

        # Merge existing storeys ----------------------------------------------------------------------------

        storeys = np.unique(trees.storey)
        mean_heights = calculate_storey_mean_heights(
            trees, set(storeys).intersection(
                (Storey.DOMINANT, Storey.UNDER, Storey.OVER)))

        dominant_storey_mean_height = mean_heights[Storey.DOMINANT]

        if Storey.UNDER in storeys:
            if abs(dominant_storey_mean_height - mean_heights[Storey.UNDER]) < 5.0:
                # Merge UNDER into DOMINANT
                trees.storey[trees.storey == Storey.UNDER] = Storey.DOMINANT

        if Storey.OVER in storeys:
            if abs(mean_heights[Storey.OVER] - dominant_storey_mean_height) < 5.0:
                # Merge OVER into DOMINANT
                trees.storey[trees.storey == Storey.OVER] = Storey.DOMINANT

        return unit, cd

    return wrapper
