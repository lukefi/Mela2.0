import numpy as np
from lukefi.metsi.data.vector_model import ReferenceTrees as VectorReferenceTrees, TreeStrata as VectorTreeStrata
from lukefi.metsi.forestry.forestry_utils import (
    generate_diameter_threshold,
)
STRATUM_SUPPLEMENT = 1
INITIAL_TREE_SUPPLEMENT = 2
SAME_TREE_DIAMETER_SUPPLEMENT = 3
SAME_TREE_D13_AGE_SUPPLEMENT = 4


def supplement_age_for_reference_trees(
    reference_trees: VectorReferenceTrees,
    stratums: VectorTreeStrata,
) -> None:
    """
    SoA-based version of `supplement_age_for_reference_trees`.

    Operates in-place on ReferenceTrees / TreeStrata vector-model containers.

    Logic mirrors the old AoS implementation:

      1) Trees that need supplementing:
         - breast_height_age is missing (NaN)
         - height > 1.3 m

      2) Strategy priority:
         - STRATUM_SUPPLEMENT:
             from strata with breast_height_age > 0 and mean_diameter > 0,
             same species as tree, choosing the best diameter match
             using the same threshold logic as in AoS implementation.
         - INITIAL_TREE_SUPPLEMENT:
             from another tree with breast_height_age > 0 and same species.
         - SAME_TREE_DIAMETER_SUPPLEMENT:
             local rule: bha = 2 * d13, bio = 9 + 2 * d13
         - SAME_TREE_D13_AGE_SUPPLEMENT:
             local rule: bha = 2 * d13, bio = bha + 9

    Only trees that initially had no `breast_height_age` are touched.
    """

    n_trees = reference_trees.size
    if n_trees == 0 or stratums.size == 0:
        return

    # Shorthands for tree fields
    t_species = reference_trees.species          # int codes, -1 == missing
    t_height = reference_trees.height           # float, NaN == missing
    t_diameter = reference_trees.breast_height_diameter
    t_bha = reference_trees.breast_height_age
    t_bio = reference_trees.biological_age

    # Shorthands for stratum fields
    s_species = stratums.species
    s_diameter = stratums.mean_diameter
    s_bha = stratums.breast_height_age
    s_bio = stratums.biological_age

    # Trees that already have a valid age (used as donors for INITIAL_TREE_SUPPLEMENT)
    age_tree_mask = (~np.isnan(t_bha)) & (t_bha > 0.0)

    # Strata that can donate age
    age_stratum_mask = (~np.isnan(s_bha)) & (s_bha > 0.0)
    strata_have_diameter = (~np.isnan(s_diameter)) & (s_diameter > 0.0)

    # Trees that need supplementing:
    # - breast_height_age is NaN
    # - height > 1.3 m
    no_age_mask = np.isnan(t_bha) & (~np.isnan(t_height)) & (t_height > 1.3)
    no_age_indices = np.nonzero(no_age_mask)[0]

    # Strategy per tree: 0 == no strategy
    strategy = np.zeros(n_trees, dtype=np.int8)
    # For INITIAL_TREE_SUPPLEMENT, we store the donor tree index
    tree_source_index = np.full(n_trees, -1, dtype=int)

    # --------- Solve strategies (mirrors AoS solve_supplement_strategy) ----------
    for i in no_age_indices:
        tree_sp = t_species[i]
        has_species = tree_sp != -1  # -1 is "missing" sentinel in vector data

        chosen_strategy = 0

        # 1) STRATUM_SUPPLEMENT: use strata with same species, age and diameter
        if has_species:
            same_sp_mask = s_species == tree_sp
            candidate_mask = age_stratum_mask & strata_have_diameter & same_sp_mask
            if np.any(candidate_mask):
                chosen_strategy = STRATUM_SUPPLEMENT

        # 2) INITIAL_TREE_SUPPLEMENT: use other trees with age & same species
        if chosen_strategy == 0 and has_species:
            same_sp_age_tree_mask = age_tree_mask & (t_species == tree_sp)
            donor_indices = np.nonzero(same_sp_age_tree_mask)[0]
            if donor_indices.size > 0:
                chosen_strategy = INITIAL_TREE_SUPPLEMENT
                # AoS version effectively uses the first matching tree
                tree_source_index[i] = int(donor_indices[0])

        # 3) Final strategies based on local information
        if chosen_strategy == 0:
            has_bio_age = (~np.isnan(t_bio[i])) & (t_bio[i] > 0.0)
            has_height_over_130 = (~np.isnan(t_height[i])) & (t_height[i] > 1.3)

            if not has_bio_age:
                chosen_strategy = SAME_TREE_DIAMETER_SUPPLEMENT
            elif not has_height_over_130:
                chosen_strategy = SAME_TREE_D13_AGE_SUPPLEMENT

        if chosen_strategy == 0:
            # Same behaviour as AoS: unsolved strategy is considered an error
            raise UserWarning(
                f"error: supplement strategy for tree index {i} (species={tree_sp}) can not be solved"
            )

        strategy[i] = chosen_strategy

    # --------- Perform supplementing in-place (mirrors perform_supplementing) ----
    for i in no_age_indices:
        s = strategy[i]
        if s == 0:
            continue

        if s == STRATUM_SUPPLEMENT:
            tree_sp = t_species[i]
            if tree_sp == -1:
                continue

            same_sp_mask = s_species == tree_sp
            candidate_mask = age_stratum_mask & strata_have_diameter & same_sp_mask
            candidate_indices = np.nonzero(candidate_mask)[0]
            if candidate_indices.size == 0:
                continue

            tree_d = t_diameter[i]
            if np.isnan(tree_d) or tree_d <= 0.0:
                continue

            # Same diameter matching rule as forestry_utils.override_from_diameter /
            # find_matching_stratum_by_diameter, but SoA-based.
            associated = int(candidate_indices[0])
            for j in candidate_indices[1:]:
                d1 = s_diameter[associated]
                d2 = s_diameter[j]
                if np.isnan(d2) or d2 <= 0.0:
                    continue
                threshold = generate_diameter_threshold(float(d1), float(d2))
                if threshold > tree_d:
                    associated = int(j)

            reference_trees.breast_height_age[i] = s_bha[associated]
            reference_trees.biological_age[i] = s_bio[associated]

        elif s == INITIAL_TREE_SUPPLEMENT:
            src = tree_source_index[i]
            if src == -1:
                continue
            reference_trees.breast_height_age[i] = t_bha[src]
            reference_trees.biological_age[i] = t_bio[src]

        elif s == SAME_TREE_DIAMETER_SUPPLEMENT:
            d = t_diameter[i]
            if np.isnan(d) or d <= 0.0:
                continue
            bha = 2.0 * d
            reference_trees.breast_height_age[i] = bha
            reference_trees.biological_age[i] = 9.0 + 2.0 * d

        elif s == SAME_TREE_D13_AGE_SUPPLEMENT:
            d = t_diameter[i]
            if np.isnan(d) or d <= 0.0:
                continue
            bha = 2.0 * d
            reference_trees.breast_height_age[i] = bha
            reference_trees.biological_age[i] = bha + 9.0
