from typing import Callable

import numpy as np
import numpy.typing as npt

from lukefi.metsi.data.enums.internal import CONIFEROUS_SPECIES, DECIDUOUS_SPECIES, Storey, TreeManagementCategory, TreeSpecies
from lukefi.metsi.data.vector_model import ReferenceTrees


def stand_has_storey(trees: ReferenceTrees, storey: Storey) -> bool:
    return bool(np.any(trees.storey == storey))


def stand_has_only_retention_storey(trees: ReferenceTrees) -> bool:
    return bool(np.all(trees.storey == Storey.RETENTION))


def stand_has_only_seeding_tree_storey(trees: ReferenceTrees) -> bool:
    return bool(np.all(
        (trees.storey == Storey.OVER) & (trees.management_category == TreeManagementCategory.SEEDING_TREE)))


def stand_has_indeterminate_storeys(trees: ReferenceTrees) -> bool:
    return bool(np.any(np.isin(trees.storey, (Storey.INDETERMINATE, Storey.UNSET))))


def stand_has_only_indeterminate_storeys(trees: ReferenceTrees) -> bool:
    return bool(np.all(np.isin(trees.storey, (Storey.INDETERMINATE, Storey.UNSET))))


def calc_tree_basal_areas(diameters: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.pi * (diameters / 200) ** 2


def calc_storey_basal_area(tree_basal_areas: npt.NDArray[np.float64], stems_per_ha: npt.NDArray[np.float64]) -> float:
    return np.sum(tree_basal_areas * stems_per_ha)


def should_use_ba_for_storey(diameters: npt.NDArray[np.float64], basal_areas: npt.NDArray[np.float64]) -> bool:
    storey_mean_diameter = np.sum(diameters * basal_areas) / np.sum(basal_areas)
    return bool(storey_mean_diameter >= 8.0)


def determine_dominant_species(tree_diameters: npt.NDArray[np.float64],
                               tree_stems_per_ha: npt.NDArray[np.float64],
                               tree_species: npt.NDArray[np.int32],
                               tree_basal_areas: npt.NDArray[np.float64]) -> TreeSpecies:

    if should_use_ba_for_storey(tree_diameters, tree_basal_areas):
        # use basal area for comparing
        storey_basal_area = calc_storey_basal_area(tree_basal_areas, tree_stems_per_ha)
        if storey_basal_area < 1.0:  # 1 m^2/ha # TODO: Check limit
            return TreeSpecies.TREELESS

        coniferous_mask = np.isin(tree_species, CONIFEROUS_SPECIES)
        coniferous_diameters = tree_diameters[coniferous_mask]
        coniferous_stems_per_ha = tree_stems_per_ha[coniferous_mask]
        coniferous_ba = np.pi * np.sum(((coniferous_diameters / 200) ** 2) * coniferous_stems_per_ha)

        if coniferous_ba >= storey_basal_area / 2:
            # coniferous dominated storey
            species_list = CONIFEROUS_SPECIES
        else:
            # deciduous dominated storey
            species_list = DECIDUOUS_SPECIES

        species_bas = {species: calc_storey_basal_area(
            tree_basal_areas[tree_species == species], tree_stems_per_ha[tree_species == species])
            for species in species_list}

        # Species with maximum basal area
        return max(species_bas, key=lambda key: species_bas[key])

    else:
        # use stems per ha for comparing
        storey_stems_per_ha = np.sum(tree_stems_per_ha)
        if storey_stems_per_ha < 400.0:  # 400 1/ha # TODO: Check limit
            return TreeSpecies.TREELESS

        coniferous_mask = np.isin(tree_species, CONIFEROUS_SPECIES)
        coniferous_stems_per_ha = tree_stems_per_ha[coniferous_mask]

        if coniferous_stems_per_ha >= storey_stems_per_ha / 2:
            # coniferous dominated storey
            species_list = CONIFEROUS_SPECIES
        else:
            # deciduous dominated storey
            species_list = DECIDUOUS_SPECIES

        species_stems = {species: np.sum(tree_stems_per_ha[tree_species == species]) for species in species_list}

        # Species with maximum stems per ha
        return max(species_stems, key=lambda key: species_stems[key])


def promote_under_or_over_storey_to_dominant(trees: ReferenceTrees):
    under_storey_mask = trees.storey == Storey.UNDER
    over_storey_mask = (trees.storey == Storey.OVER) & (
        trees.management_category != TreeManagementCategory.SEEDING_TREE)  # Exclude seeding trees

    under_storey_tree_diameters = trees.breast_height_diameter[under_storey_mask]
    under_storey_tree_basal_areas = trees.basal_area[under_storey_mask]
    under_storey_tree_species = trees.species[under_storey_mask]
    under_storey_tree_stems_per_ha = trees.stems_per_ha[under_storey_mask]

    over_storey_tree_diameters = trees.breast_height_diameter[over_storey_mask]
    over_storey_tree_basal_areas = trees.basal_area[over_storey_mask]
    over_storey_tree_species = trees.species[over_storey_mask]
    over_storey_tree_stems_per_ha = trees.stems_per_ha[over_storey_mask]

    compare_under_storey_by_ba = should_use_ba_for_storey(
        under_storey_tree_diameters,
        under_storey_tree_basal_areas)
    compare_over_storey_by_ba = should_use_ba_for_storey(
        over_storey_tree_diameters,
        over_storey_tree_basal_areas
    )

    if compare_under_storey_by_ba != compare_over_storey_by_ba:
        # Can not compare UNDER and OVER storeys, promote UNDER
        trees.storey[under_storey_mask] = Storey.DOMINANT
        return

    under_storey_dominant_species = determine_dominant_species(under_storey_tree_diameters,
                                                               under_storey_tree_stems_per_ha,
                                                               under_storey_tree_species,
                                                               under_storey_tree_basal_areas)
    over_storey_dominant_species = determine_dominant_species(over_storey_tree_diameters,
                                                              over_storey_tree_stems_per_ha,
                                                              over_storey_tree_species,
                                                              over_storey_tree_basal_areas)

    if compare_under_storey_by_ba:  # and compare_over_storey_by_ba
        # Compare by basal area
        under_storey_dominant_species_mask = under_storey_tree_species == under_storey_dominant_species
        over_storey_dominant_species_mask = over_storey_tree_species == over_storey_dominant_species

        under_storey_dominant_species_ba = calc_storey_basal_area(
            trees.basal_area[under_storey_dominant_species_mask],
            trees.stems_per_ha[under_storey_dominant_species_mask])
        over_storey_dominant_species_ba = calc_storey_basal_area(
            trees.basal_area[over_storey_dominant_species_mask],
            trees.stems_per_ha[over_storey_dominant_species_mask])

        if under_storey_dominant_species_ba >= over_storey_dominant_species_ba:
            # Promote UNDER
            trees.storey[under_storey_mask] = Storey.DOMINANT
            return
        else:
            # Promote OVER
            trees.storey[over_storey_mask] = Storey.DOMINANT
            return

    else:
        # Compare by stems
        under_storey_dominant_species_mask = under_storey_tree_species == under_storey_dominant_species
        over_storey_dominant_species_mask = over_storey_tree_species == over_storey_dominant_species

        under_storey_dominant_species_stems = trees.stems_per_ha[under_storey_dominant_species_mask]
        over_storey_dominant_species_stems = trees.stems_per_ha[over_storey_dominant_species_mask]

        if under_storey_dominant_species_stems >= over_storey_dominant_species_stems:
            # Promote UNDER
            trees.storey[under_storey_mask] = Storey.DOMINANT
            return
        else:
            # Promote OVER
            trees.storey[over_storey_mask] = Storey.DOMINANT
            return


def promote_either_storey_to_dominant(trees: ReferenceTrees,
                                      mask1: npt.NDArray[np.bool_],
                                      mask2: npt.NDArray[np.bool_],
                                      fallback: Callable[[ReferenceTrees], None]):

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
        storey_1_tree_basal_areas)
    compare_storey_2_by_ba = should_use_ba_for_storey(
        storey_2_tree_diameters,
        storey_2_tree_basal_areas
    )

    if compare_storey_1_by_ba != compare_storey_2_by_ba:
        # Can not compare storeys, use fallback
        fallback(trees)
        return

    storey_1_dominant_species = determine_dominant_species(storey_1_tree_diameters,
                                                           storey_1_tree_stems_per_ha,
                                                           storey_1_tree_species,
                                                           storey_1_tree_basal_areas)
    storey_2_dominant_species = determine_dominant_species(storey_2_tree_diameters,
                                                           storey_2_tree_stems_per_ha,
                                                           storey_2_tree_species,
                                                           storey_2_tree_basal_areas)

    if compare_storey_1_by_ba:  # and compare_over_storey_by_ba
        # Compare by basal area
        storey_1_dominant_species_mask = storey_1_tree_species == storey_1_dominant_species
        storey_2_dominant_species_mask = storey_2_tree_species == storey_2_dominant_species

        storey_1_dominant_species_ba = calc_storey_basal_area(
            trees.basal_area[storey_1_dominant_species_mask],
            trees.stems_per_ha[storey_1_dominant_species_mask])
        storey_2_dominant_species_ba = calc_storey_basal_area(
            trees.basal_area[storey_2_dominant_species_mask],
            trees.stems_per_ha[storey_2_dominant_species_mask])

        if storey_1_dominant_species_ba >= storey_2_dominant_species_ba:
            # Promote storey 1
            trees.storey[mask1] = Storey.DOMINANT
            return
        else:
            # Promote storey 2
            trees.storey[mask2] = Storey.DOMINANT
            return

    else:
        # Compare by stems
        storey_1_dominant_species_mask = storey_1_tree_species == storey_1_dominant_species
        storey_2_dominant_species_mask = storey_2_tree_species == storey_2_dominant_species

        storey_1_dominant_species_stems = trees.stems_per_ha[storey_1_dominant_species_mask]
        storey_2_dominant_species_stems = trees.stems_per_ha[storey_2_dominant_species_mask]

        if storey_1_dominant_species_stems >= storey_2_dominant_species_stems:
            # Promote storey 1
            trees.storey[mask1] = Storey.DOMINANT
            return
        else:
            # Promote storey 2
            trees.storey[mask2] = Storey.DOMINANT
            return


def promote_remote_or_removal_storey_to_dominant(trees: ReferenceTrees):
    remote_storey_mask = trees.storey == Storey.REMOTE
    removal_storey_mask = trees.storey == Storey.REMOVAL
