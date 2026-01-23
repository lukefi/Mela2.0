""" Module contains tree generation logic that uses distribution based tree generation models
(see. distributions module) """
from enum import Enum
import numpy as np
from lukefi.metsi.data.enums.internal import TreeSpecies
from lukefi.metsi.data.model import ReferenceTree, TreeStratum
from lukefi.metsi.forestry.preprocessing import distributions
from lukefi.metsi.forestry.preprocessing.naslund import naslund_height, naslund_correction
from lukefi.metsi.forestry.preprocessing.tree_generation_lm import tree_generation_lm
from lukefi.metsi.forestry.forestry_utils import (
    find_matching_storey_stratum_for_tree,
)
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import (
    ReferenceTrees as VectorReferenceTrees, TreeStrata as VectorTreeStrata, DTYPES_TREE
)


class TreeStrategy(Enum):
    WEIBULL_DISTRIBUTION = 'weibull_distribution'
    LM_TREES = 'LM_TREES'
    HEIGHT_DISTRIBUTION = 'HEIGHT_DISTRIBUTION'
    SKIP = 'skip_tree_generation'


def finalize_trees(reference_trees: list[ReferenceTree], stratum: TreeStratum) -> list[ReferenceTree]:
    """ For all given trees inflates the common variables from stratum. """
    stratum.number_of_generated_trees = len(reference_trees)
    for i, reference_tree in enumerate(reference_trees):
        reference_tree.stand = stratum.stand
        reference_tree.species = stratum.species
        reference_tree.origin = stratum.origin
        reference_tree.breast_height_age = 0.0 \
            if stratum.number_of_generated_trees == 1 else stratum.get_breast_height_age()
        reference_tree.biological_age = stratum.biological_age
        if reference_tree.breast_height_age == 0.0 and (reference_tree.breast_height_diameter or 0.0) > 0.0:
            reference_tree.breast_height_age = 1.0
        reference_tree.tree_number = i + 1
        reference_tree.stems_per_ha = None if reference_tree.stems_per_ha is None \
            else round(reference_tree.stems_per_ha, 2)
        reference_tree.breast_height_diameter = None if reference_tree.breast_height_diameter is None \
            else round(reference_tree.breast_height_diameter, 2)
        reference_tree.height = None if reference_tree.height is None \
            else round(reference_tree.height, 2)
    return reference_trees


def trees_from_weibull(stratum: TreeStratum, **params) -> list[ReferenceTree]:
    """ Generate N trees from weibull distribution.

    For a single tree, stem count and diameter are obtained
    from weibull distribution.
    The height is derived with Näslund height prediction model.
    """
    # stems_per_ha and diameter
    result = distributions.weibull(
        params.get('n_trees', 0),
        stratum.mean_diameter or 0.0,
        stratum.basal_area or 0.0,
        stratum.mean_height or 0.0)
    # height
    for reference_tree in result:
        height = naslund_height(
            reference_tree.breast_height_diameter,
            stratum.species)
        reference_tree.height = 0.0 if height is None else height
    # height correction
    h_scalar = naslund_correction(stratum.species or TreeSpecies.PINE,
                                  stratum.mean_diameter or 0.0,
                                  stratum.mean_height or 0.0)
    for reference_tree in result:
        reference_tree.height = round((h_scalar or 0.0) * (reference_tree.height or 0.0), 2)

    return result


def trees_from_sapling_height_distribution(stratum: TreeStratum, **params) -> list[ReferenceTree]:
    """  Generate N trees from height distribution """
    return distributions.sapling_height_distribution(
        stratum,
        0.0,
        params.get('n_trees', 0))


def solve_tree_generation_strategy(stratum: TreeStratum, method='weibull') -> str:
    """ Solves the strategy of tree generation for given stratum """

    if stratum.has_height_over_130_cm():
        # big trees
        if stratum.has_diameter() and stratum.has_height() and stratum.has_basal_area() and method == 'weibull':
            return TreeStrategy.WEIBULL_DISTRIBUTION.value
        if not stratum.sapling_stratum and stratum.has_diameter() \
                and (stratum.has_basal_area() or stratum.has_stems_per_ha()) and method == 'lm':
            return TreeStrategy.LM_TREES.value
        if stratum.has_diameter() and stratum.has_height() and stratum.has_stems_per_ha():
            return TreeStrategy.HEIGHT_DISTRIBUTION.value
        return TreeStrategy.SKIP.value
    # small trees
    if stratum.has_height() and stratum.sapling_stratum:
        return TreeStrategy.HEIGHT_DISTRIBUTION.value
    return TreeStrategy.SKIP.value


def reference_trees_from_tree_stratum(stratum: TreeStratum, **params) -> list[ReferenceTree]:
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
    method = params.get("method", "weibull")
    strategy = solve_tree_generation_strategy(stratum, method)

    if strategy == TreeStrategy.HEIGHT_DISTRIBUTION.value:
        result = trees_from_sapling_height_distribution(stratum, **params)

    elif strategy == TreeStrategy.WEIBULL_DISTRIBUTION.value:
        result = trees_from_weibull(stratum, **params)

    elif strategy == TreeStrategy.LM_TREES.value:
        # Make mypy/pylance happy and be explicit about missing stand info
        stand = stratum.stand
        if stand is None:
            raise ValueError(
                f"LM_TREES strategy requires 'stratum.stand' to be set "
                f"(stratum {stratum.identifier})"
            )

        degree_days = float(stand.degree_days or 0.0)
        basal_area = float(stand.basal_area or 0.0)

        result = tree_generation_lm(stratum, degree_days, basal_area, **params)

    elif strategy == TreeStrategy.SKIP.value:
        print(f"\nStratum {stratum.identifier} has no height or diameter usable for generating trees")
        return []

    else:
        raise UserWarning(f"Unable to generate reference trees from stratum {stratum.identifier}")

    # Filter out tiny / zero-stem trees
    result = [rt for rt in result if round(rt.stems_per_ha or 0.0, 2) > 0.0]

    return finalize_trees(result, stratum)


def _generate_trees_for_stratum(
    stand: ForestStand,
    strata_vec: VectorTreeStrata,
    s_idx: int,
    measured_trees_vec: VectorReferenceTrees,
    measured_indices: list[int],
    **params,
) -> list[ReferenceTree]:
    """
    For a single stratum row in SoA, generate AoS ReferenceTree instances.
    This function is the only place where we temporarily create AoS dataclasses.
    """

    # Build a tiny AoS TreeStratum for this row (just for internal use)
    s = TreeStratum()
    s.stand = stand
    s.identifier = str(strata_vec.identifier[s_idx])
    s.species = TreeSpecies(int(strata_vec.species[s_idx])) if strata_vec.species[s_idx] != -1 else None
    s.origin = int(strata_vec.origin[s_idx]) if strata_vec.origin[s_idx] != -1 else None
    s.stems_per_ha = float(strata_vec.stems_per_ha[s_idx]) if not np.isnan(strata_vec.stems_per_ha[s_idx]) else None
    s.mean_diameter = float(strata_vec.mean_diameter[s_idx]) if not np.isnan(strata_vec.mean_diameter[s_idx]) else None
    s.mean_height = float(strata_vec.mean_height[s_idx]) if not np.isnan(strata_vec.mean_height[s_idx]) else None
    s.breast_height_age = float(
        strata_vec.breast_height_age[s_idx]) if not np.isnan(
        strata_vec.breast_height_age[s_idx]) else None
    s.biological_age = float(
        strata_vec.biological_age[s_idx]) if not np.isnan(
        strata_vec.biological_age[s_idx]) else None
    s.basal_area = float(strata_vec.basal_area[s_idx]) if not np.isnan(strata_vec.basal_area[s_idx]) else None
    s.sapling_stems_per_ha = float(
        strata_vec.sapling_stems_per_ha[s_idx]) if not np.isnan(
        strata_vec.sapling_stems_per_ha[s_idx]) else None
    s.sapling_stratum = bool(strata_vec.sapling_stratum[s_idx])
    s.storey = None  # or convert from int if you keep Storey in SoA
    # etc. for fields you actually use in tree_generation

    # Attach measured trees (AoS) ONLY for LM usage
    if measured_indices:
        source_trees: list[ReferenceTree] = []
        for t_idx in measured_indices:
            t = ReferenceTree()
            t.stand = stand
            t.identifier = str(measured_trees_vec.identifier[t_idx])
            t.species = TreeSpecies(int(measured_trees_vec.species[t_idx]))
            t.breast_height_diameter = float(measured_trees_vec.breast_height_diameter[t_idx])
            t.height = float(
                measured_trees_vec.height[t_idx]) if not np.isnan(
                measured_trees_vec.height[t_idx]) else None
            t.measured_height = float(
                measured_trees_vec.measured_height[t_idx]) if not np.isnan(
                measured_trees_vec.measured_height[t_idx]) else None
            t.stems_per_ha = float(
                measured_trees_vec.stems_per_ha[t_idx]) if not np.isnan(
                measured_trees_vec.stems_per_ha[t_idx]) else None
            t.storey = None  # convert if needed
            t.tuhon_ilmiasu = str(measured_trees_vec.tuhon_ilmiasu[t_idx]) or None
            source_trees.append(t)
        setattr(s, "_trees", source_trees)

    # Now reuse the existing AoS logic:
    trees = reference_trees_from_tree_stratum(s, **params)
    return trees


def _associate_measured_trees_to_strata(
    strata_vec: VectorTreeStrata,
    trees_vec: VectorReferenceTrees,
    diameter_threshold: float,
) -> dict[int, list[int]]:
    """
    Return mapping: stratum_index -> list of measured_tree_indices
    Reimplements find_matching_storey_stratum_for_tree but on SoA arrays.
    """
    result: dict[int, list[int]] = {}

    for t_idx in range(trees_vec.size):
        s_idx = find_matching_storey_stratum_for_tree(
            tree_index=t_idx,
            trees=trees_vec,
            strata=strata_vec,
            diameter_threshold=diameter_threshold,
        )
        if s_idx is not None:
            result.setdefault(s_idx, []).append(t_idx)

    return result


def generate_reference_trees(
    stand: ForestStand,
    *,
    strata_vec: VectorTreeStrata | None = None,
    measured_trees_vec: VectorReferenceTrees | None = None,
    **params,
) -> VectorReferenceTrees:
    """
    SoA entry point: generate reference trees for all strata of a stand.

    - Reads from stand.tree_strata / stand.reference_trees (or overrides).
    - Applies the same strategy logic as reference_trees_from_tree_stratum, but per-row.
    - Returns a fresh ReferenceTrees SoA container with generated trees.
    """

    if strata_vec is None:
        strata_vec = stand.tree_strata
    if measured_trees_vec is None:
        measured_trees_vec = stand.reference_trees

    # We'll accumulate attr-dicts and vectorize once at the end for speed
    attr_dict: dict[str, list] = {name: [] for name in DTYPES_TREE}

    # Optional: pre-compute any stand-wide info (like basal_area) if needed
    # For LM we need stand.degree_days and stand.basal_area

    # For LM we also need to know which measured trees "belong" to each stratum.
    # We'll do that by indices, not AoS:
    stratum_to_measured_indices = _associate_measured_trees_to_strata(
        strata_vec, measured_trees_vec, params.get("stratum_association_diameter_threshold", 2.5)
    )

    start_idx = int(stand.reference_trees.size)
    for s_idx in range(strata_vec.size):
        # Skip empty / invalid strata early if you want
        trees_for_stratum = _generate_trees_for_stratum(
            stand,
            strata_vec,
            s_idx,
            measured_trees_vec,
            stratum_to_measured_indices.get(s_idx, []),
            **params,
        )

        for t in trees_for_stratum:
            if not t.identifier:
                # Use a stand-wide running index, not per-stratum local_idx
                identifier = f"{stand.identifier}-{start_idx + 1}-tree"
                t.identifier = identifier
                start_idx += 1

            attr_dict["identifier"].append(t.identifier or "")
            attr_dict["tree_number"].append(t.tree_number or 0)
            attr_dict["species"].append(int(t.species) if t.species is not None else -1)
            attr_dict["breast_height_diameter"].append(
                t.breast_height_diameter if t.breast_height_diameter is not None else np.nan
            )
            attr_dict["height"].append(t.height if t.height is not None else np.nan)
            attr_dict["measured_height"].append(
                t.measured_height if t.measured_height is not None else np.nan
            )
            attr_dict["breast_height_age"].append(
                t.breast_height_age if t.breast_height_age is not None else np.nan
            )
            attr_dict["biological_age"].append(
                t.biological_age if t.biological_age is not None else np.nan
            )
            attr_dict["stems_per_ha"].append(
                t.stems_per_ha if t.stems_per_ha is not None else np.nan
            )
            attr_dict["origin"].append(t.origin if t.origin is not None else -1)

            # The rest: fill with neutral defaults so lengths stay consistent
            attr_dict["management_category"].append(-1)
            attr_dict["saw_log_volume_reduction_factor"].append(np.nan)
            attr_dict["pruning_year"].append(0)
            attr_dict["age_when_10cm_diameter_at_breast_height"].append(-1)
            attr_dict["stand_origin_relative_position"].append(
                (0.0, 0.0, 0.0)
            )
            attr_dict["lowest_living_branch_height"].append(np.nan)
            attr_dict["tree_category"].append("")  # or "reference", up to you
            attr_dict["storey"].append(-1 if t.storey is None else int(t.storey))
            attr_dict["sapling"].append(False)  # generated trees are usually non-sapling
            attr_dict["tree_type"].append("")   # set if you have semantics for this
            attr_dict["tuhon_ilmiasu"].append(t.tuhon_ilmiasu or "")
            attr_dict["basal_area"].append(np.nan)
            attr_dict["volume"].append(np.nan)

    vec = VectorReferenceTrees()
    vec.vectorize(attr_dict)

    stand.tree_strata = VectorTreeStrata()

    return vec
