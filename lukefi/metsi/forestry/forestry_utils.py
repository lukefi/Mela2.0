import math
from collections.abc import Callable, Iterable
from typing import Optional
import numpy as np
from lukefi.metsi.data.enums.internal import TreeSpecies
from lukefi.metsi.data.model import ReferenceTree, TreeStratum
from lukefi.metsi.data.vector_model import (
    ReferenceTrees as VectorReferenceTrees,
    TreeStrata as VectorTreeStrata,
)


def calculate_basal_area(tree: ReferenceTree) -> float:
    """ Single reference tree basal area calculation.

    The tree should contain breast height diameter (in cm) and stesm per hectare for the species spesific calculations.

    :param tree: Single ReferenceTree instance with breast height diameter (in cm) and stems per hectare properties.
    :return reference tree basal area in square meters (m^2)
    """
    meters_factor = 0.01
    radius = (tree.breast_height_diameter or 0.0) * 0.5 * meters_factor
    single_basal_area = math.pi * math.pow(radius, 2)
    return (single_basal_area or 0.0) * (tree.stems_per_ha or 0.0)


def generate_diameter_threshold(d1: float, d2: float) -> float:
    """ Threshold value for diameter based comparison of two stratums.

    Threshold will have a value based on relative distance of at most 50% of the distance between d[0] and d[1].
    """
    greater = max((d1, d2))
    lesser = min((d1, d2))
    return greater + (lesser - greater) * (greater / (lesser + greater))


def override_from_diameter(initial_stratum: TreeStratum, candidate_stratum: TreeStratum,
                           reference_tree: ReferenceTree) -> TreeStratum:
    """ Out of given strata, return the stratum for which the mean diameter better matches the reference tree diameter.
    This happens by calculating a threshold value based on which of the stratum diameters
    is greater and comparing the threshold to reference tree diameter.

    :param initial_stratum: Stratum which is assumed as the current match for the reference tree
    :param candidate_stratum: Stratum which is tested for better compatiblity than the initial stratum
    :param reference_tree: The tree for which the supplementing will be done

    :returns: the better matching stratum
    """
    threshold = generate_diameter_threshold(
        initial_stratum.mean_diameter or 0.0,
        candidate_stratum.mean_diameter or 0.0)
    if not threshold or not reference_tree.breast_height_diameter:
        return initial_stratum
    if threshold > reference_tree.breast_height_diameter:
        return candidate_stratum
    return initial_stratum


def find_matching_stratum_by_diameter(
        reference_tree: ReferenceTree,
        strata: list[TreeStratum]) -> Optional[TreeStratum]:
    """ Solves from which stratum the supplementing of reference tree should happen.

    Return a stratum that has the closest diameter to the reference tree diameter.

    :param reference_tree: A reference tree for which a matching stratum needs to be found
    :param strata: list of stratums from which a diameter match is searched
    :returns: a matching stratum or None if not match is possible
    """
    if len(strata) == 0:
        return None
    associated_stratum = strata[0]
    for stratum in strata[1:]:
        if stratum.has_diameter():
            associated_stratum = override_from_diameter(associated_stratum, stratum, reference_tree)
    return associated_stratum


def find_matching_stratum_by_diameter_lm(
        reference_tree: ReferenceTree,
        strata: Iterable[TreeStratum],
        threshold=2.5) -> Optional[TreeStratum]:
    """
    Find the stratum that has the closest diameter to the reference tree diameter by factor of difference, where the
    reference tree diameter is between the stratum mean diameter divided by threshold and multiplied by threshold.

    :param reference_tree: candidate reference tree
    :param strata: candidate strata
    :param threshold: threshold factor for diameter bounds
    :return: matching stratum or None if no match is found
    """
    candidate = min(
        strata,
        key=lambda stratum: abs(reference_tree.breast_height_diameter or 0.0 / (stratum.mean_diameter or 0.0) - 1),
        default=None
    )
    if candidate is None:
        return None

    lower = candidate.mean_diameter or 0.0 / threshold
    upper = candidate.mean_diameter or 0.0 * threshold
    if not lower or not upper or not reference_tree.breast_height_diameter:
        return None
    if lower < reference_tree.breast_height_diameter < upper:
        return candidate
    return None


def split_list_by_predicate(items: list, predicate: Callable) -> tuple[list, list]:
    """ Splits a list into two lists based on a predicate.

    :param items: list to be split
    :param predicate: Predicate used to split the list
    :return: Tuple of lists, where the first list contains the items that match the predicate and the second list
        contains the items that do not match the predicate.
    """
    matching_items = []
    non_matching_items = []
    for item in items:
        if predicate(item):
            matching_items.append(item)
        else:
            non_matching_items.append(item)
    return matching_items, non_matching_items


def find_strata_by_similar_species(species: TreeSpecies, strata: list[TreeStratum]) -> list[TreeStratum]:
    """
    Find a list of strata which have a similar species to the given species.
    Out of deciduous trees, silver birch is considered most similar to downy birch and vice versa.
    """
    candidates: list[TreeStratum] = []

    if species.is_deciduous():
        if species == TreeSpecies.DOWNY_BIRCH:
            candidates.extend(
                filter(lambda s: s.species == TreeSpecies.SILVER_BIRCH, strata)
            )
        elif species == TreeSpecies.SILVER_BIRCH:
            candidates.extend(
                filter(lambda s: s.species == TreeSpecies.DOWNY_BIRCH, strata)
            )
        else:
            candidates.extend(
                filter(
                    lambda s: s.species is not None and s.species.is_deciduous(),
                    strata,
                )
            )
    elif species.is_coniferous():
        candidates.extend(
            filter(
                lambda s: s.species is not None and s.species.is_coniferous(),
                strata,
            )
        )

    return candidates


def find_matching_storey_stratum_for_tree(
    tree_index: int,
    trees: VectorReferenceTrees,
    strata: VectorTreeStrata,
    diameter_threshold: float = 2.5,
) -> Optional[int]:
    """
    SoA-based version of `find_matching_storey_stratum_for_tree`.

    Parameters
    ----------
    tree_index:
        Index of the tree in `trees` whose matching stratum we want.
    trees:
        SoA container of reference trees (VectorReferenceTrees).
    strata:
        SoA container of strata (VectorTreeStrata).
    diameter_threshold:
        Threshold factor used for diameter matching. Same semantics as
        `find_matching_stratum_by_diameter_lm`:
        tree_d is accepted if:
            mean_diameter / threshold < tree_d < mean_diameter * threshold

    Returns
    -------
    Optional[int]
        Index of the matching stratum in `strata`, or None if no match.
    """

    # Basic sanity
    if strata.size == 0 or trees.size == 0:
        return None
    if tree_index < 0 or tree_index >= trees.size:
        return None

    tree_storey = trees.storey[tree_index]
    tree_species_code = trees.species[tree_index]
    tree_diameter = trees.breast_height_diameter[tree_index]

    # If no valid diameter, we can't do diameter-based matching
    if np.isnan(tree_diameter) or tree_diameter <= 0.0:
        return None

    # 1) Strata in the same storey
    same_storey_idx = np.nonzero(strata.storey == tree_storey)[0]
    if same_storey_idx.size == 0:
        return None

    same_storey_species = strata.species[same_storey_idx]

    # 2) Split same-storey strata into same-species and other-species
    same_species_mask = same_storey_species == tree_species_code
    same_species_idx = same_storey_idx[same_species_mask]
    other_species_idx = same_storey_idx[~same_species_mask]

    # Helper to build "similar species" indices (SoA version of find_strata_by_similar_species)
    def _similar_species_indices() -> np.ndarray:
        # If tree species is missing, just bail out
        if tree_species_code == -1:
            return np.array([], dtype=int)

        try:
            tree_species = TreeSpecies(tree_species_code)
        except ValueError:
            return np.array([], dtype=int)

        # No other strata to compare against
        if other_species_idx.size == 0:
            return np.array([], dtype=int)

        result: list[int] = []
        for j in other_species_idx:
            s_code = strata.species[j]
            if s_code == -1:
                continue
            try:
                s_species = TreeSpecies(s_code)
            except ValueError:
                continue

            if tree_species.is_deciduous():
                # Special birch handling
                if (
                    tree_species == TreeSpecies.DOWNY_BIRCH
                    and s_species == TreeSpecies.SILVER_BIRCH
                ):
                    result.append(int(j))
                elif (
                    tree_species == TreeSpecies.SILVER_BIRCH
                    and s_species == TreeSpecies.DOWNY_BIRCH
                ):
                    result.append(int(j))
                elif s_species.is_deciduous():
                    result.append(int(j))
            elif tree_species.is_coniferous():
                if s_species.is_coniferous():
                    result.append(int(j))

        return np.array(result, dtype=int)

    # 3) Candidate strata selection (mirror AoS logic):
    #    - prefer same species
    #    - otherwise, similar species (deciduous/coniferous rules)
    #    - otherwise, any same-storey strata
    if same_species_idx.size > 0:
        candidate_idx = same_species_idx
    else:
        similar_idx = _similar_species_indices()
        if similar_idx.size > 0:
            candidate_idx = similar_idx
        else:
            candidate_idx = same_storey_idx

    if candidate_idx.size == 0:
        return None

    # 4) Filter to strata that "have diameter" (SoA equivalent of TreeStratum.has_diameter)
    candidate_mean_d = strata.mean_diameter[candidate_idx]
    has_diameter_mask = (~np.isnan(candidate_mean_d)) & (candidate_mean_d > 0.0)
    candidate_idx = candidate_idx[has_diameter_mask]
    candidate_mean_d = candidate_mean_d[has_diameter_mask]

    if candidate_idx.size == 0:
        return None

    # 5) Diameter-based selection and threshold check (SoA version of find_matching_stratum_by_diameter_lm)
    #    - pick stratum i that minimizes abs(tree_d / mean_d_i - 1)
    #    - accept only if tree_d within [mean_d / threshold, mean_d * threshold]
    ratios = np.abs(tree_diameter / candidate_mean_d - 1.0)
    best_pos = int(np.argmin(ratios))
    best_idx = int(candidate_idx[best_pos])
    best_d = float(candidate_mean_d[best_pos])

    lower = best_d / diameter_threshold
    upper = best_d * diameter_threshold

    if lower < tree_diameter < upper:
        return best_idx

    return None
