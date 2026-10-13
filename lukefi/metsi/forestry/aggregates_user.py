import numpy as np
import numpy.typing as npt

from lukefi.metsi.data.enums.internal import TreeSpecies
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.forestry.aggregates_internal import (
    calc_arithmetic_mean,
    calc_ba_weighted_mean,
    calc_tree_basal_areas as calc_tree_basal_areas_internal,
    calc_basal_area as calc_basal_area_internal,
    calc_mean_biological_age as calc_mean_biological_age_internal,
    calc_dominant_height as calc_dominant_height_internal,
    calc_mean_height as calc_mean_height_internal,
    determine_dominant_species as determine_dominant_species_internal
)


def calc_tree_basal_areas(trees: ReferenceTrees, mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    return calc_tree_basal_areas_internal(trees.breast_height_diameter[mask])


def calc_basal_area(trees: ReferenceTrees, mask: npt.NDArray[np.bool_]) -> float:
    return calc_basal_area_internal(trees.basal_area[mask], trees.stems_per_ha[mask])


def calc_stems_per_ha(trees: ReferenceTrees, mask: npt.NDArray[np.bool_]) -> float:
    return np.sum(trees.stems_per_ha[mask])


def calc_mean_biological_age(trees: ReferenceTrees, mask: npt.NDArray[np.bool_]) -> float | None:
    return calc_mean_biological_age_internal(trees.biological_age[mask],
                                             trees.stems_per_ha[mask],
                                             trees.basal_area[mask],
                                             trees.breast_height_diameter[mask])


def calc_dominant_height(trees: ReferenceTrees, mask: npt.NDArray[np.bool_]) -> float | None:
    return calc_dominant_height_internal(trees.stems_per_ha[mask],
                                         trees.breast_height_diameter[mask],
                                         trees.height[mask])


def calc_mean_height(trees: ReferenceTrees, mask: npt.NDArray[np.bool_]) -> float | None:
    return calc_mean_height_internal(trees, mask)


def determine_dominant_species(trees: ReferenceTrees, mask: npt.NDArray[np.bool_]) -> TreeSpecies:
    return determine_dominant_species_internal(trees.breast_height_diameter[mask],
                                               trees.stems_per_ha[mask],
                                               trees.species[mask],
                                               trees.basal_area[mask])


def calc_ba_weighted_mean_diameter(trees: ReferenceTrees, mask: npt.NDArray[np.bool_]) -> float | None:
    return calc_ba_weighted_mean(trees.breast_height_diameter[mask],
                                 trees.stems_per_ha[mask],
                                 trees.basal_area[mask])


def calc_ba_weighted_mean_height(trees: ReferenceTrees, mask: npt.NDArray[np.bool_]) -> float | None:
    return calc_ba_weighted_mean(trees.height[mask],
                                 trees.stems_per_ha[mask],
                                 trees.basal_area[mask])


def calc_ba_weighted_mean_biological_age(trees: ReferenceTrees, mask: npt.NDArray[np.bool_]) -> float | None:
    return calc_ba_weighted_mean(trees.biological_age[mask],
                                 trees.stems_per_ha[mask],
                                 trees.basal_area[mask])


def calc_arithmetic_mean_diameter(trees: ReferenceTrees, mask: npt.NDArray[np.bool_]) -> float | None:
    return calc_arithmetic_mean(trees.breast_height_diameter[mask],
                                trees.stems_per_ha[mask])


def calc_arithmetic_mean_height(trees: ReferenceTrees, mask: npt.NDArray[np.bool_]) -> float | None:
    return calc_arithmetic_mean(trees.height[mask],
                                trees.stems_per_ha[mask])


def calc_arithmetic_mean_biological_age(trees: ReferenceTrees, mask: npt.NDArray[np.bool_]) -> float | None:
    return calc_arithmetic_mean(trees.biological_age[mask],
                                trees.stems_per_ha[mask])
