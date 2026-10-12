from lukefi.metsi.data.enums.internal import CONIFEROUS_SPECIES, DECIDUOUS_SPECIES, TreeSpecies
from lukefi.metsi.data.vector_model import ReferenceTrees

import numpy as np
import numpy.typing as npt


BA_LIMIT = 1.0  # m^2/ha
STEMS_LIMIT = 400.0  # 1/ha


def calc_tree_basal_areas(diameters: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.pi * (diameters / 200) ** 2


def calc_basal_area(tree_basal_areas: npt.NDArray[np.float64], stems_per_ha: npt.NDArray[np.float64]) -> float:
    return np.sum(tree_basal_areas * stems_per_ha)


def should_use_ba_as_weight(diameters: npt.NDArray[np.float64],
                            basal_areas: npt.NDArray[np.float64],
                            stems_per_ha: npt.NDArray[np.float64]) -> bool:
    if np.sum(basal_areas) == 0:
        return False

    storey_mean_diameter = np.sum(diameters * basal_areas * stems_per_ha) / np.sum(basal_areas * stems_per_ha)
    return bool(storey_mean_diameter >= 8.0)


def calc_mean_biological_age(biological_ages: npt.NDArray[np.float64],
                             stems: npt.NDArray[np.float64],
                             basal_areas: npt.NDArray[np.float64],
                             diameters: npt.NDArray[np.float64]) -> float | None:
    if not np.any(stems > 0):
        return None

    if should_use_ba_as_weight(diameters, basal_areas, stems):
        return np.sum(biological_ages * basal_areas * stems) / np.sum(basal_areas * stems)

    return np.sum(biological_ages * stems) / np.sum(stems)


def calc_dominant_height(trees: ReferenceTrees, mask: npt.NDArray[np.bool_]) -> float | None:
    if not np.any(mask):
        return None

    stems = trees.stems_per_ha[mask]
    diameter = trees.breast_height_diameter[mask]
    height = trees.height[mask]

    sorting = np.flip(np.argsort(diameter))
    sorted_stems = stems[sorting]
    sorted_height = height[sorting]
    cumulative_sorted_stems = np.cumsum(sorted_stems)

    i_100_largest_arr = np.flatnonzero(cumulative_sorted_stems >= 100)
    if len(i_100_largest_arr) == 0:
        stems_smallest: float = sorted_stems[-1]
        i_100_largest: int = int(np.count_nonzero(mask) - 1)
    elif i_100_largest_arr[0] == 0:
        stems_smallest = 100.0
        i_100_largest = 0
    else:
        i_100_largest = i_100_largest_arr[0]
        stems_smallest = 100 - cumulative_sorted_stems[i_100_largest - 1]

    numerator_1 = np.sum(sorted_stems[:i_100_largest] * sorted_height[:i_100_largest])
    numerator_2: float = stems_smallest * sorted_height[i_100_largest]
    denominator = min(100, cumulative_sorted_stems[i_100_largest])

    return (numerator_1 + numerator_2) / denominator


def calc_mean_height(trees: ReferenceTrees, mask: npt.NDArray[np.bool_]) -> float:
    if not np.any(mask):
        return 0.0

    tree_stems_per_ha = trees.stems_per_ha[mask]

    if not np.any(tree_stems_per_ha > 0):
        return 0.0

    tree_diameters = trees.breast_height_diameter[mask]
    tree_basal_areas = trees.basal_area[mask]
    tree_heights = trees.height[mask]

    should_use_ba = should_use_ba_as_weight(tree_diameters, tree_basal_areas, tree_stems_per_ha)
    if should_use_ba:
        return np.sum(tree_basal_areas * tree_stems_per_ha * tree_heights) / \
            np.sum(tree_basal_areas * tree_stems_per_ha)
    return np.sum(tree_heights * tree_stems_per_ha) / np.sum(tree_stems_per_ha)


def determine_dominant_species(tree_diameters: npt.NDArray[np.float64],
                               tree_stems_per_ha: npt.NDArray[np.float64],
                               tree_species: npt.NDArray[np.int32],
                               tree_basal_areas: npt.NDArray[np.float64]) -> TreeSpecies:

    if should_use_ba_as_weight(tree_diameters, tree_basal_areas, tree_stems_per_ha):
        # Use basal area for comparing
        storey_basal_area = calc_basal_area(tree_basal_areas, tree_stems_per_ha)
        if storey_basal_area < BA_LIMIT:
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

        species_bas = {species: calc_basal_area(
            tree_basal_areas[tree_species == species], tree_stems_per_ha[tree_species == species])
            for species in species_list}

        # Species with maximum basal area
        return max(species_bas, key=lambda key: species_bas[key])

    # Use stems per ha for comparing
    storey_stems_per_ha = np.sum(tree_stems_per_ha)
    if storey_stems_per_ha < STEMS_LIMIT:
        return TreeSpecies.TREELESS

    coniferous_mask = np.isin(tree_species, CONIFEROUS_SPECIES)
    coniferous_total_stems_per_ha = np.sum(tree_stems_per_ha[coniferous_mask])

    if coniferous_total_stems_per_ha >= storey_stems_per_ha / 2:
        # Coniferous dominated storey
        species_list = CONIFEROUS_SPECIES
    else:
        # Deciduous dominated storey
        species_list = DECIDUOUS_SPECIES

    species_stems = {species: np.sum(tree_stems_per_ha[tree_species == species]) for species in species_list}

    # Species with maximum stems per ha
    return max(species_stems, key=lambda key: species_stems[key])


def calc_ba_weighted_mean(quantity: npt.NDArray[np.float64],
                          tree_stems: npt.NDArray[np.float64],
                          basal_areas: npt.NDArray[np.float64]) -> float | None:
    ba = np.sum(tree_stems * basal_areas)
    if ba <= 0:
        return None
    return np.sum(tree_stems * basal_areas * quantity) / ba


def calc_arithmetic_mean(quantity: npt.NDArray[np.float64],
                         tree_stems: npt.NDArray[np.float64]) -> float | None:
    stems = np.sum(tree_stems)
    if stems <= 0:
        return None
    return np.sum(tree_stems * quantity) / stems
