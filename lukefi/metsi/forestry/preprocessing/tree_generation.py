""" Module contains tree generation logic that uses distribution based tree generation models
(see. distributions module) """
from enum import StrEnum
from typing import Optional

import numpy as np
import numpy.typing as npt
from lukefi.metsi.data.enums.internal import LandUseCategory, Storey, StratumRank, TreeManagementCategory, TreeSpecies
from lukefi.metsi.data.enums.vmi import VmiIteration
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStratum
from lukefi.metsi.forestry.preprocessing import distributions
from lukefi.metsi.forestry.preprocessing.height import predict_tree_height
from lukefi.metsi.forestry.preprocessing.naslund import naslund_height, naslund_correction
from lukefi.metsi.forestry.preprocessing.ages import ages
from lukefi.metsi.forestry.preprocessing.tree_generation_lm import determine_hmalli_value, tree_generation_lm


class TreeStrategy(StrEnum):
    WEIBULL_DISTRIBUTION = 'weibull_distribution'
    LM_TREES = 'LM_TREES'
    HEIGHT_DISTRIBUTION = 'HEIGHT_DISTRIBUTION'
    SKIP = 'skip_tree_generation'


def _finalize_trees(reference_trees: ReferenceTrees, stratum: TreeStratum, ng_scale: float) -> ReferenceTrees:
    """ For all given trees inflates the common variables from stratum. """
    n_trees = len(reference_trees)
    stratum.number_of_generated_trees = n_trees

    for i in range(n_trees):
        reference_trees.tree_number[i] = i + 1

    reference_trees.species[np.isin(reference_trees.species, (TreeSpecies.UNKNOWN,
                                    TreeSpecies.UNSET, TreeSpecies.TREELESS))] = stratum.species

    big_trees = reference_trees.height > 1.3
    reference_trees.breast_height_age[big_trees] = max(stratum.get_breast_height_age(), 1)
    reference_trees.breast_height_age[~big_trees] = 0.0

    reference_trees.biological_age.fill(stratum.biological_age)

    reference_trees.stems_per_ha = np.round(ng_scale * reference_trees.stems_per_ha, 2)

    reference_trees.breast_height_diameter = np.round(reference_trees.breast_height_diameter, 2)

    reference_trees.breast_height_diameter[big_trees] = np.maximum(
        reference_trees.breast_height_diameter[big_trees], 0.1)

    reference_trees.height = np.round(reference_trees.height, 2)

    retained = stratum.stratum_rank == StratumRank.RETENTION_TREE_STOREY
    reference_trees.management_category.fill(TreeManagementCategory.RETENTION_TREE if retained else 1)
    reference_trees.storey.fill(Storey.SPARE if retained else stratum.storey)

    reference_trees.origin.fill(stratum.origin)
    reference_trees.stratum.fill(stratum.stratum_number)

    return reference_trees


def _trees_from_weibull(stratum: TreeStratum, n_trees: int) -> ReferenceTrees:
    """ Generate N trees from weibull distribution.

    For a single tree, stem count and diameter are obtained
    from weibull distribution.
    The height is derived with Näslund height prediction model.
    """
    # stems_per_ha and diameter
    result = distributions.weibull(n_trees, stratum.mean_diameter, stratum.basal_area or 0.0, stratum.mean_height)

    # height
    for i in range(len(result)):
        height = naslund_height(result.breast_height_diameter[i], stratum.species)
        result.height[i] = 0.0 if height is None else height

    # height correction
    h_scalar = naslund_correction(stratum.species, stratum.mean_diameter, stratum.mean_height)
    for i in range(len(result)):
        result.height[i] = round(h_scalar * result.height[i], 2)

    return result


def _trees_from_sapling_height_distribution(stratum: TreeStratum, n_trees: int) -> ReferenceTrees:
    """  Generate N trees from height distribution """
    return distributions.sapling_height_distribution(stratum, 0.0, n_trees)


def _solve_tree_generation_strategy(stand: ForestStand, stratum: TreeStratum, method='weibull') -> TreeStrategy:
    """ Solves the strategy of tree generation for given stratum """

    if method == 'lm' and stratum.stratum_rank in (
            StratumRank.NON_ESTABLISHED_SEEDLINGS,
            StratumRank.DAMAGED_TREE_STRATUM):
        return TreeStrategy.SKIP

    if stratum.mean_height > 1.3 or stratum.mean_diameter > 2:
        # big trees
        if (stratum.mean_diameter > 0.0 and stratum.mean_height >
                0.0 and stratum.basal_area is not None and stratum.basal_area > 0.0 and method == 'weibull'):
            return TreeStrategy.WEIBULL_DISTRIBUTION

        if stand.land_use_category == LandUseCategory.SCRUB_LAND and stratum.basal_area is not None and \
                stratum.basal_area > 0.0 and method == 'lm':
            return TreeStrategy.LM_TREES

        if all([
            stratum.basal_area == 0.0,
            stratum.stems_per_ha > 0.0,
            2.0 > stratum.mean_height > 0.0,
            method == 'lm'
        ]):
            return TreeStrategy.HEIGHT_DISTRIBUTION

        if stratum.mean_diameter > 0.0 and stratum.basal_area is not None and stratum.basal_area >= 0.0 and \
                method == 'lm':
            return TreeStrategy.LM_TREES

        if stratum.mean_height > 0.0 and stratum.stems_per_ha > 0.0:
            return TreeStrategy.HEIGHT_DISTRIBUTION

        return TreeStrategy.SKIP

    # small trees
    if stratum.mean_height > 0.0 and stratum.stems_per_ha > 0.0:
        return TreeStrategy.HEIGHT_DISTRIBUTION

    return TreeStrategy.SKIP


def reference_trees_from_tree_stratum(stand: ForestStand, stratum: TreeStratum, **params) -> Optional[ReferenceTrees]:
    """ Composes N number of reference trees based on values of the stratum.

    The tree generation strategies: weibull distribution, lm_trees and height distribution.
    For big trees generation strategies are weibull or lm_trees depending on configuration, and height distributions.
    Small trees (height < 1.3 meters) are generated with height distribution.

    Big trees need diameter (cm), height (m) and basal area or stem count for the generation process to succeed.
    Small trees need only height (m) and sapling stem count.
    All other cases are skipped.

    :param stratum: Single stratum instance.
    :return: list of reference trees derived from given stratum.
    """
    result: ReferenceTrees
    strategy = _solve_tree_generation_strategy(stand, stratum, params.get('method', 'weibull'))

    if strategy == TreeStrategy.HEIGHT_DISTRIBUTION:
        result = _trees_from_sapling_height_distribution(stratum, params["n_trees"])
        if params.get("scale_height_distribution_stems_by_ba", True):
            assert stratum.basal_area is not None
            if result.breast_height_diameter.any():
                result.stems_per_ha = result.stems_per_ha * stratum.basal_area / \
                    _calculate_basal_area_from_trees(result.stems_per_ha, result.breast_height_diameter)

    elif strategy == TreeStrategy.WEIBULL_DISTRIBUTION:
        result = _trees_from_weibull(stratum, params["n_trees"])

    elif strategy == TreeStrategy.LM_TREES:
        assert stand.degree_days is not None
        assert stand.basal_area is not None
        result = tree_generation_lm(stand, stratum, **params)

    elif strategy == TreeStrategy.SKIP:
        print(f"\nStratum {stratum.identifier} has no height or diameter usable for generating trees")
        return None

    else:
        raise UserWarning(f"Unable to generate reference trees from stratum {stratum.identifier}")

    enough_stems = result.stems_per_ha > 0.005
    result = result[enough_stems]

    return _finalize_trees(result, stratum, params.get('ng_scale_factor', 1))


def _calculate_basal_area_from_trees(stems_per_ha, breast_height_diameter) -> float:
    return np.pi * np.sum(stems_per_ha * ((breast_height_diameter / 200) ** 2))


def _determine_ages(stand: ForestStand,
                    new_trees: ReferenceTrees,
                    retention_trees_mask: npt.NDArray[np.bool_],
                    tree_i: int,
                    added_years: float) -> tuple[float, float]:
    trees = stand.reference_trees
    if trees.biological_age[tree_i] > 0:
        return trees.breast_height_age[tree_i], trees.biological_age[tree_i]
    return ages(stand, new_trees + trees[retention_trees_mask], trees.get_tree(tree_i), added_years)


def adjust_retention_trees(stand: ForestStand,
                           new_trees: ReferenceTrees,
                           retention_trees_mask: npt.NDArray[np.bool_],
                           nfi_iteration: VmiIteration):
    # Scales the stem counts so that basal area does not increase
    # Basal area may increse if the basal area of the retention trees is greater than
    # basal area of the reference trees

    trees = stand.reference_trees

    g_retention = _calculate_basal_area_from_trees(
        trees.stems_per_ha[retention_trees_mask],
        trees.breast_height_diameter[retention_trees_mask])

    g_generated_not_retention_trees = _calculate_basal_area_from_trees(
        new_trees.stems_per_ha[new_trees.management_category != TreeManagementCategory.RETENTION_TREE],
        new_trees.breast_height_diameter[new_trees.management_category != TreeManagementCategory.RETENTION_TREE])

    scale_factor_stand = max((g_generated_not_retention_trees - g_retention) / g_generated_not_retention_trees, 0) \
        if g_generated_not_retention_trees > 0.0 else 1

    # skaalauksessa jätetään kuhunkin ositteeseen min(1, ositteen kuvauaspuiden  ppa) m2/ha
    # säästöpuite ei skaalata
    itree = -1
    for i_stratum in range(len(stand.tree_strata)):
        scale_factor_stratum = scale_factor_stand

        g_stratum = 0
        g_stratum_retention = 0
        itree0 = itree
        for i in range(stand.tree_strata.number_of_generated_trees[i_stratum]):
            itree = itree + 1
            g_stratum = g_stratum + new_trees.stems_per_ha[itree] * \
                np.pi * ((new_trees.breast_height_diameter[itree] / 200)**2)
            if new_trees.management_category[itree] == TreeManagementCategory.RETENTION_TREE:
                g_stratum_retention = g_stratum_retention + \
                    new_trees.stems_per_ha[itree] * np.pi * ((new_trees.breast_height_diameter[itree] / 200)**2)
        g_stratum_scaled = scale_factor_stratum * g_stratum

        if g_stratum_scaled < 1 and g_stratum <= 1:
            scale_factor_stratum = 1
        elif g_stratum_scaled < 1 < g_stratum:
            if g_stratum_retention == 0:
                scale_factor_stratum = 1 / g_stratum
            elif g_stratum_retention < 1:
                scale_factor_stratum = (1 - g_stratum_retention) / g_stratum
            else:
                scale_factor_stratum = 0

        for i in range(stand.tree_strata.number_of_generated_trees[i_stratum]):
            itree = itree0 + i + 1
            if new_trees.management_category[itree] != TreeManagementCategory.RETENTION_TREE:
                new_trees.stems_per_ha[itree] = scale_factor_stratum * new_trees.stems_per_ha[itree]

    stand_tree_count = len(new_trees)

    for j, i in enumerate(np.where(retention_trees_mask)[0]):
        trees.identifier[i] = f"{stand.identifier}-{stand_tree_count + j + 1}-tree"
        trees.management_category[i] = TreeManagementCategory.RETENTION_TREE
        trees.storey[i] = Storey.SPARE
        breast_height_age, biological_age = _determine_ages(stand, new_trees, retention_trees_mask, i, 10)
        trees.breast_height_age[i] = breast_height_age
        trees.biological_age[i] = biological_age
        if np.isnan(trees.height[i]) or trees.height[i] == 0:
            trees.height[i] = predict_tree_height(
                nfi_iteration,
                determine_hmalli_value(
                    TreeSpecies(
                        trees.species[i])),
                stand.degree_days or 0.0,
                float(trees.breast_height_diameter[i]),
                float(trees.breast_height_diameter[i]),
                1.0
            )


def adjust_ages(stand: ForestStand, trees: ReferenceTrees):
    # adjust tree ages (from age model) by subtracting the difference between
    # basal area weighted age of trees and age of stratum

    itree = -1
    strata = stand.tree_strata
    for i in range(len(strata)):
        if strata.number_of_generated_trees[i] > 0:
            g = 0
            agesum = 0
            itreeages = itree
            for _ in range(strata.number_of_generated_trees[i]):
                itree = itree + 1
                if trees.stems_per_ha[itree] > 0:
                    gtree = trees.stems_per_ha[itree] * np.pi * ((trees.breast_height_diameter[itree] / 200)**2)
                    agesum = agesum + gtree * trees.breast_height_age[itree]
                    g = g + gtree

            mean_age = agesum / g if g > 0 else 0

            breast_height_age = 0.0
            if strata.breast_height_age[i] > 0.0:
                breast_height_age = strata.breast_height_age[i]
            elif strata.biological_age[i] > 0.0:
                breast_height_age = max(strata.biological_age[i] - 12.0, 0.0)
            else:
                breast_height_age = 0.0

            age_diff = mean_age - breast_height_age

            if age_diff != 0:
                for _ in range(strata.number_of_generated_trees[i]):
                    itreeages = itreeages + 1
                    if trees.height[itreeages] > 1.3:
                        trees.breast_height_age[itreeages] = max(trees.breast_height_age[itreeages] - age_diff, 1)
                        trees.biological_age[itreeages] = max(trees.biological_age[itreeages] - age_diff, 1)
