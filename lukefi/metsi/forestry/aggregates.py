from lukefi.metsi.data.vector_model import ReferenceTrees

import numpy as np
import numpy.typing as npt


def calc_basal_area(tree_basal_areas: npt.NDArray[np.float64], stems_per_ha: npt.NDArray[np.float64]) -> float:
    return np.sum(tree_basal_areas * stems_per_ha)


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
