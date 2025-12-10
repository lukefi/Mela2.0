""" Tests suites for forestryfunctions.preprocessing.* modules """
import unittest
from collections import namedtuple

from lukefi.metsi.data.model import TreeStratum, ReferenceTree, ForestStand
from lukefi.metsi.data.enums.internal import TreeSpecies
from lukefi.metsi.forestry.preprocessing import tree_generation


class TestTreeGeneration(unittest.TestCase):
    """
    These tests exercise the *AoS* core logic in tree_generation:

    - trees_from_weibull
    - finalize_trees
    - solve_tree_generation_strategy
    - reference_trees_from_tree_stratum

    The SoA helpers (VectorReferenceTrees / TreeStrata) are tested elsewhere;
    here we focus on the per-stratum logic that SoA wrappers delegate to.
    """

    # Helper for building AoS TreeStratum from simple tuples, like the old test
    Input = namedtuple(
        "Input",
        "species diameter basal_area mean_height "
        "breast_height_age biological_age stand stems_per_ha sapling_stems_per_ha",
    )

    def create_test_stratums(self, inputs):
        strata = []
        for i in inputs:
            s = TreeStratum()
            s.species = i.species
            s.mean_diameter = i.diameter
            s.basal_area = i.basal_area
            s.mean_height = i.mean_height
            s.breast_height_age = i.breast_height_age
            s.biological_age = i.biological_age
            s.stand = i.stand
            s.stems_per_ha = i.stems_per_ha
            s.sapling_stems_per_ha = i.sapling_stems_per_ha
            # emulate old behaviour: sapling_stratum flag from sapling stems
            s.sapling_stratum = (
                False
                if i.sapling_stems_per_ha is None or i.sapling_stems_per_ha == 0
                else True
            )
            strata.append(s)
        return strata

    # ------------------------------------------------------------------
    # trees_from_weibull
    # ------------------------------------------------------------------
    def test_trees_from_weibull(self):
        """
        trees_from_weibull should:
        - return exactly n_trees ReferenceTree instances
        - fill in positive diameter / height / stems_per_ha
        - honour the stratum's species for height model
        """
        fixture = TreeStratum()
        fixture.mean_diameter = 28.0
        fixture.basal_area = 27.0
        fixture.mean_height = 22.0
        fixture.species = TreeSpecies.PINE

        n_trees = 10
        result = tree_generation.trees_from_weibull(fixture, n_trees=n_trees)

        self.assertEqual(n_trees, len(result))
        # Spot-check the first tree: numbers should be sensible and positive.
        t0 = result[0]
        self.assertIsInstance(t0, ReferenceTree)
        self.assertGreater(t0.breast_height_diameter, 0.0)
        self.assertGreater(t0.height, 0.0)
        self.assertGreater(t0.stems_per_ha, 0.0)

    # ------------------------------------------------------------------
    # finalize_trees
    # ------------------------------------------------------------------
    def test_finalize_trees(self):
        """
        finalize_trees should:
        - propagate stand and species from stratum
        - derive breast_height_age via TreeStratum.get_breast_height_age()
        - copy biological_age
        - set tree_number 1..N
        - update stratum.number_of_generated_trees
        """
        n_trees = 2
        stratum = TreeStratum()
        stratum.species = TreeSpecies.PINE
        stratum.stand = "0-012-001-01-1"
        stratum.breast_height_age = 25.0
        stratum.biological_age = 32.0

        reference_trees = [ReferenceTree() for _ in range(n_trees)]
        # give some values so rounding branches are exercised
        reference_trees[0].breast_height_diameter = 10.123
        reference_trees[0].height = 8.456
        reference_trees[0].stems_per_ha = 100.789

        result = tree_generation.finalize_trees(reference_trees, stratum)

        self.assertEqual(stratum.number_of_generated_trees, n_trees)

        for idx, t in enumerate(result, start=1):
            self.assertEqual(t.stand, stratum.stand)
            self.assertEqual(t.species, stratum.species)
            # For multiple trees, breast_height_age should come from helper
            self.assertEqual(t.breast_height_age, stratum.get_breast_height_age())
            self.assertEqual(t.biological_age, stratum.biological_age)
            self.assertEqual(t.tree_number, idx)

        # Check rounding occurred on the modified tree
        t0 = result[0]
        self.assertEqual(t0.breast_height_diameter, round(10.123, 2))
        self.assertEqual(t0.height, round(8.456, 2))
        self.assertEqual(t0.stems_per_ha, round(100.789, 2))

    # ------------------------------------------------------------------
    # solve_tree_generation_strategy
    # ------------------------------------------------------------------
    def test_solve_tree_generation_strategy(self):
        """
        Check that solve_tree_generation_strategy picks the correct *string*
        strategy code given the stratum state.

        NOTE: tree_generation.solve_tree_generation_strategy now returns
        TreeStrategy.<...>.value (a string), not the enum member itself.
        """
        stand = ForestStand()
        stand.identifier = "stand-1"

        stratum_inputs = [
            # Big trees, full weibull info: diameter, height, basal_area
            # -> WEIBULL_DISTRIBUTION
            self.Input(
                None, 10.0, 33.0, 8.0, None, None, stand, None, None
            ),
            # Big trees, diameter + height + stems_per_ha, but no basal_area
            # -> HEIGHT_DISTRIBUTION (for big trees)
            self.Input(
                None, 10.0, None, 8.0, None, None, stand, 55.0, None
            ),
            # Small trees, sapling stratum, height > 0, sapling stems > 0
            # -> HEIGHT_DISTRIBUTION (sapling height distribution)
            self.Input(
                None, None, None, 1.2, None, None, stand, None, 111.0
            ),
            # No usable diameter/height/stems -> SKIP
            self.Input(
                None, None, None, None, None, None, stand, None, None
            ),
        ]

        strata = self.create_test_stratums(stratum_inputs)

        expected_values = [
            tree_generation.TreeStrategy.WEIBULL_DISTRIBUTION.value,
            tree_generation.TreeStrategy.HEIGHT_DISTRIBUTION.value,
            tree_generation.TreeStrategy.HEIGHT_DISTRIBUTION.value,
            tree_generation.TreeStrategy.SKIP.value,
        ]

        for stratum, expected in zip(strata, expected_values):
            result = tree_generation.solve_tree_generation_strategy(stratum)
            self.assertEqual(expected, result)

        # LM_TREES extra sanity check: big trees + method="lm"
        lm_stratum = TreeStratum()
        lm_stratum.mean_diameter = 10.0
        lm_stratum.mean_height = 8.0
        lm_stratum.basal_area = 20.0
        lm_stratum.sapling_stratum = False
        lm_result = tree_generation.solve_tree_generation_strategy(
            lm_stratum, method="lm"
        )
        self.assertEqual(
            tree_generation.TreeStrategy.LM_TREES.value, lm_result
        )

    # ------------------------------------------------------------------
    # reference_trees_from_tree_stratum (integration-style)
    # ------------------------------------------------------------------
    def test_reference_trees_from_tree_stratum_integration(self):
        """
        Integration test: from a stratum, we should get:
        - N generated trees for a valid big-tree weibull case
        - [] for a completely unusable stratum (SKIP)
        - finalize_trees is applied (stand/species/ages set)
        """
        stand = ForestStand()
        stand.identifier = "0-023-002-02-1"

        # 1) Big trees -> WEIBULL_DISTRIBUTION -> trees_from_weibull
        big_stratum = TreeStratum()
        big_stratum.stand = stand
        big_stratum.species = TreeSpecies.PINE
        big_stratum.mean_diameter = 28.0
        big_stratum.basal_area = 27.0
        big_stratum.mean_height = 22.0
        big_stratum.breast_height_age = 15.0
        big_stratum.biological_age = 16.0

        # 2) Completely empty -> SKIP
        empty_stratum = TreeStratum()
        empty_stratum.stand = stand
        empty_stratum.species = TreeSpecies.PINE

        n_trees = 10

        # Big tree case
        big_result = tree_generation.reference_trees_from_tree_stratum(
            big_stratum, n_trees=n_trees
        )

        self.assertEqual(n_trees, len(big_result))
        for t in big_result:
            self.assertEqual(t.stand, stand)
            self.assertEqual(t.species, TreeSpecies.PINE)
            # breast_height_age and biological_age propagated
            self.assertEqual(t.breast_height_age, big_stratum.get_breast_height_age())
            self.assertEqual(t.biological_age, big_stratum.biological_age)
            # stems_per_ha should be positive and rounded
            self.assertGreater(t.stems_per_ha, 0.0)

        # Skip case
        skip_result = tree_generation.reference_trees_from_tree_stratum(
            empty_stratum, n_trees=n_trees
        )
        self.assertEqual(0, len(skip_result))
