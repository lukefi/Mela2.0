""" Module contains tree generation logic that uses distribution based tree generation models
(see. distributions module) """
from enum import StrEnum
from typing import Optional
from lukefi.metsi.data.enums.internal import Storey, TreeSpecies
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStratum
from lukefi.metsi.forestry.preprocessing import distributions
from lukefi.metsi.forestry.preprocessing.naslund import naslund_height, naslund_correction
from lukefi.metsi.forestry.preprocessing.tree_generation_lm import tree_generation_lm


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
        # reference_tree.stand = stratum.stand
        reference_trees.species[i] = stratum.species if reference_trees.species[i] in (
            TreeSpecies.UNKNOWN, TreeSpecies.UNSET, TreeSpecies.TREELESS) else reference_trees.species[i]
        reference_trees.breast_height_age[i] = max(stratum.get_breast_height_age(), 1) if \
            reference_trees.height[i] > 1.3 else 0.0
        reference_trees.biological_age[i] = stratum.biological_age
        reference_trees.tree_number[i] = i + 1
        reference_trees.stems_per_ha[i] = round(ng_scale * reference_trees.stems_per_ha[i], 2)
        reference_trees.breast_height_diameter[i] = round(reference_trees.breast_height_diameter[i], 2)
        if reference_trees.height[i] > 1.3:
            reference_trees.breast_height_diameter[i] = max(reference_trees.breast_height_diameter[i], 0.1)
        reference_trees.height[i] = round(reference_trees.height[i], 2)
        retained = stratum.asema == 3
        reference_trees.management_category[i] = 2 if retained else 1
        reference_trees.storey[i] = Storey.SPARE if retained else stratum.storey
        reference_trees.origin[i] = stratum.origin

    return reference_trees


def _trees_from_weibull(stratum: TreeStratum, n_trees: int) -> ReferenceTrees:
    """ Generate N trees from weibull distribution.

    For a single tree, stem count and diameter are obtained
    from weibull distribution.
    The height is derived with Näslund height prediction model.
    """
    # stems_per_ha and diameter
    result = distributions.weibull(n_trees, stratum.mean_diameter, stratum.basal_area, stratum.mean_height)

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

    if method == 'lm' and stratum.asema in (7, 8):
        return TreeStrategy.SKIP

    if stratum.mean_height > 1.3:
        # big trees
        if (stratum.mean_diameter > 0.0 and stratum.mean_height >
                0.0 and stratum.basal_area > 0.0 and method == 'weibull'):
            return TreeStrategy.WEIBULL_DISTRIBUTION
        if stand.land_use_category == 2 and stratum.basal_area > 0.0 and method == 'lm':
            return TreeStrategy.LM_TREES
        if all([
            stratum.basal_area == 0.0,
            stratum.stems_per_ha > 0.0,
            2.0 > stratum.mean_height > 0.0,
            method == 'lm'
        ]):
            return TreeStrategy.HEIGHT_DISTRIBUTION
        if stratum.mean_diameter > 0.0 and stratum.basal_area >= 0.0 and method == 'lm':
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
