import unittest
from collections import namedtuple

import numpy as np

from lukefi.metsi.forestry.preprocessing import age_supplementing as age_sup
from lukefi.metsi.data.vector_model import (
    ReferenceTrees as VectorReferenceTrees,
    TreeStrata as VectorTreeStrata,
)


Input = namedtuple("Input", "id species diameter breast_height_age biological_age height")


def create_vector_trees(inputs: list[Input]) -> VectorReferenceTrees:
    """
    Helper: create a SoA ReferenceTrees container from a list of Input tuples.
    Only a subset of fields is populated; other fields get vector-model defaults.
    """
    attr = {
        "identifier": [i.id for i in inputs],
        "species": [i.species for i in inputs],
        "breast_height_diameter": [i.diameter for i in inputs],
        "breast_height_age": [i.breast_height_age for i in inputs],
        "biological_age": [i.biological_age for i in inputs],
        "height": [i.height for i in inputs],
    }
    vec = VectorReferenceTrees()
    vec.vectorize(attr)
    return vec


def create_vector_strata(inputs: list[Input]) -> VectorTreeStrata:
    """
    Helper: create a SoA TreeStrata container from a list of Input tuples.
    """
    attr = {
        "identifier": [i.id for i in inputs],
        "species": [i.species for i in inputs],
        "mean_diameter": [i.diameter for i in inputs],
        "breast_height_age": [i.breast_height_age for i in inputs],
        "biological_age": [i.biological_age for i in inputs],
    }
    vec = VectorTreeStrata()
    vec.vectorize(attr)
    return vec


class TestAgeSupplementingSoA(unittest.TestCase):
    def test_stratum_based_supplement(self):
        """
        Tree with missing age, height>1.3 and matching species should
        get age from the best matching stratum (STRATUM_SUPPLEMENT path).
        """
        strata_inputs = [
            Input("stratum-1", 1, 12.0, 6.0, 7.0, None),
            # Different species / zero diameter → not a candidate
            Input("stratum-2", 2, 0.0, 66.0, 77.0, None),
        ]
        tree_inputs = [
            Input("tree-1", 1, 10.0, None, None, 3.0),
        ]

        trees = create_vector_trees(tree_inputs)
        strata = create_vector_strata(strata_inputs)

        age_sup.supplement_age_for_reference_trees(trees, strata)

        self.assertAlmostEqual(trees.breast_height_age[0], 6.0)
        self.assertAlmostEqual(trees.biological_age[0], 7.0)

    def test_initial_tree_supplement(self):
        """
        If no suitable stratum is available, but another tree of the same
        species *does* have age, the target tree should copy its age
        (INITIAL_TREE_SUPPLEMENT path).
        """
        strata_inputs = [
            # Same species, but no usable age (breast_height_age None) → not a candidate
            Input("stratum-1", 1, 12.0, None, None, None),
        ]
        tree_inputs = [
            # Donor tree with age
            Input("tree-donor", 1, 10.0, 5.0, 8.0, 4.0),
            # Target tree (no age, height>1.3)
            Input("tree-target", 1, 10.0, None, None, 3.0),
        ]

        trees = create_vector_trees(tree_inputs)
        strata = create_vector_strata(strata_inputs)

        age_sup.supplement_age_for_reference_trees(trees, strata)

        # Donor unchanged
        self.assertAlmostEqual(trees.breast_height_age[0], 5.0)
        self.assertAlmostEqual(trees.biological_age[0], 8.0)

        # Target copied from donor
        self.assertAlmostEqual(trees.breast_height_age[1], 5.0)
        self.assertAlmostEqual(trees.biological_age[1], 8.0)

    def test_same_tree_diameter_supplement(self):
        """
        If no strata and no donor trees are usable, and the tree has no
        biological age, use the local diameter rule:
            bha = 2 * d13
            bio = 9 + 2 * d13
        (SAME_TREE_DIAMETER_SUPPLEMENT path).
        """
        strata_inputs = [
            # Same species, but no age → not usable as donor
            Input("stratum-1", 1, 12.0, None, None, None),
        ]
        d13 = 10.0
        tree_inputs = [
            Input("tree-1", 1, d13, None, None, 3.0),
        ]

        trees = create_vector_trees(tree_inputs)
        strata = create_vector_strata(strata_inputs)

        age_sup.supplement_age_for_reference_trees(trees, strata)

        expected_bha = 2.0 * d13
        expected_bio = 9.0 + 2.0 * d13

        self.assertAlmostEqual(trees.breast_height_age[0], expected_bha)
        self.assertAlmostEqual(trees.biological_age[0], expected_bio)

    def test_height_threshold_excludes_short_trees(self):
        """
        Trees with height <= 1.3 m must NOT be supplemented, even if they
        otherwise match; only height > 1.3 m is considered.
        """
        strata_inputs = [
            Input("stratum-1", 1, 12.0, 6.0, 7.0, None),
        ]
        tree_inputs = [
            # Height exactly 1.3 → must NOT be touched
            Input("tree-sapling", 1, 10.0, None, 2.0, 1.3),
            # Height > 1.3 → should get stratum age
            Input("tree-adult", 1, 10.0, None, 5.0, 3.5),
        ]

        trees = create_vector_trees(tree_inputs)
        strata = create_vector_strata(strata_inputs)

        age_sup.supplement_age_for_reference_trees(trees, strata)

        # Tree 0: height==1.3 → still no breast_height_age
        self.assertTrue(np.isnan(trees.breast_height_age[0]))
        # Tree 1: supplemented from stratum
        self.assertAlmostEqual(trees.breast_height_age[1], 6.0)
        self.assertAlmostEqual(trees.biological_age[1], 7.0)

    def test_unsolved_strategy_raises(self):
        """
        If a tree needs supplementing (bha missing, height>1.3) and
        neither strata nor other trees allow selecting any strategy,
        the function should raise UserWarning (matches AoS fail case).
        """
        strata_inputs = [
            # Different species / unusable data, so no stratum supplement
            Input("stratum-1", 2, 0.0, None, None, None),
        ]
        tree_inputs = [
            # Missing bha, height>1.3, but already has biological age;
            # no usable strata or donor trees → strategy remains unsolved.
            Input("tree-fail", 1, 10.0, None, 0.1, 1.31),
        ]

        trees = create_vector_trees(tree_inputs)
        strata = create_vector_strata(strata_inputs)

        with self.assertRaises(UserWarning):
            age_sup.supplement_age_for_reference_trees(trees, strata)
