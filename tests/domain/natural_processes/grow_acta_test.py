import unittest
from copy import deepcopy
import numpy as np
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from tests.test_utils import prepare_growth_test_stand


class GrowActaTest(unittest.TestCase):

    def assert_domain_sensibility(self, stand: ForestStand, before: ForestStand):
        """Sanity check: growth shouldn't move to the wrong direction.

        * stems_per_ha must never increase
        * height must never decrease
        * breast_height_diameter must never decrease

        Exact values for domain functions should be tested in the underlying library implementations.
        """
        self.assertTrue(np.all(stand.reference_trees.stems_per_ha <= before.reference_trees.stems_per_ha))
        self.assertTrue(
            np.all(stand.reference_trees.breast_height_diameter >= before.reference_trees.breast_height_diameter))
        self.assertTrue(np.all(stand.reference_trees.height >= before.reference_trees.height))

    def test_grow_acta(self):
        stand = prepare_growth_test_stand()

        # Keep a pre-growth snapshot for monotonicity checks.
        before = deepcopy(stand)

        stand, _ = grow_acta_fn(stand)
        self.assert_domain_sensibility(stand, before)
        self.assertFalse(stand.reference_trees.sapling[2])
        self.assertEqual(stand.reference_trees.biological_age[0], 60)
        self.assertEqual(stand.reference_trees.biological_age[1], 42)
        self.assertEqual(stand.reference_trees.biological_age[2], 6)
        self.assertEqual(stand.reference_trees.breast_height_age[0], 15)
        self.assertEqual(stand.reference_trees.breast_height_age[1], 15)
        self.assertEqual(stand.reference_trees.breast_height_age[2], 6)
        self.assertEqual(stand.year, 2030)
