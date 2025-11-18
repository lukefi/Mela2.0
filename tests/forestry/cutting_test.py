import unittest
from typing import cast
import numpy as np
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.forestry.harvest.cutting import cutting
from lukefi.metsi.domain.collected_data import RemovedTrees

def _all_trees(_stand: ForestStand, data: ReferenceTrees) -> np.ndarray:
    # selection function that includes every tree
    return np.repeat(True, data.size)


class CuttingTest(unittest.TestCase):
    def make_stand_with_trees(self, identifier="stand-cut", year=2030):
        stand = ForestStand()
        stand.identifier = identifier
        stand.year = year

        # Build a clean ReferenceTrees block explicitly
        rt = ReferenceTrees()
        rt.vectorize({
            "identifier": [f"{identifier}-1-tree", f"{identifier}-2-tree", f"{identifier}-3-tree"],
            "stems_per_ha": [100.0, 200.0, 300.0],
            "height": [10.0, 20.0, 30.0],
            "species": [1, 1, 2],
            "tree_number": [1, 2, 3],
        })
        stand.reference_trees = rt
        return stand

    def test_basic_cutting_uniform_profile_absolute_target(self):
        """
        With a flat profile y=0.5 and absolute target equal to 50% of total stems,
        each tree should have 50% of its stems removed.
        """
        stand = self.make_stand_with_trees()
        total = np.sum(stand.reference_trees.stems_per_ha)
        params = {
            "tree_selection": {
                "target": {"type": "absolute", "var": "stems_per_ha", "amount": 0.5 * total},
                "sets": [{
                    "sfunction": _all_trees,
                    "order_var": "height",
                    "target_var": "stems_per_ha",
                    "target_type": "absolute",
                    "target_amount": 0.5 * total,
                    "profile_x": [0.0, 1.0],
                    "profile_y": [0.5, 0.5],
                    "profile_xmode": "relative",
                }]
            },
            "mode": "odds_units",
            "select_from_all": True,
            "cutting_method": 7,
        }

        before = stand.reference_trees.stems_per_ha.copy()
        updated, cdata = cutting(stand, **params)

        self.assertEqual(1, len(cdata))
        self.assertIsInstance(cdata[0], RemovedTrees)
        rt = cast(RemovedTrees, cdata[0]).removed_trees
        np.testing.assert_allclose(rt.stems_per_ha, before * 0.5, rtol=0, atol=1e-9)
        # Bookkeeping fields set only when provided
        self.assertEqual(2030, updated.cutting_year)
        self.assertEqual(7, updated.method_of_last_cutting)

        # Expected removals: 50%, so resulting stems are halved
        expected_after = before * 0.5
        np.testing.assert_allclose(expected_after, updated.reference_trees.stems_per_ha, rtol=0, atol=1e-9)

    def test_returns_early_when_no_trees(self):
        stand = ForestStand()
        stand.identifier = "empty-stand"
        stand.year = 2040

        # Ensure the stand truly has no trees
        stand.reference_trees = ReferenceTrees()  # size == 0

        # With no trees, cutting should return early regardless of sets content
        updated, cdata = cutting(
            stand,
            tree_selection={"target": {"type": "absolute", "var": "stems_per_ha", "amount": 10.0}, "sets": []},
            mode="odds_units",
            select_from_all=True,
        )

        self.assertEqual([], cdata)
        self.assertEqual(0, updated.reference_trees.size)
        self.assertIsNone(updated.cutting_year)
        self.assertIsNone(updated.method_of_last_cutting)

    def test_validations_and_bad_inputs(self):
        stand = self.make_stand_with_trees()

        # Missing tree_selection
        with self.assertRaises(MetsiException):
            cutting(stand, mode="odds_units", select_from_all=True)

        # Empty sets
        with self.assertRaises(MetsiException):
            cutting(stand, tree_selection={"target": {"type": "absolute", "var": "stems_per_ha", "amount": 10.0},
                                           "sets": []},
                    mode="odds_units", select_from_all=True)

        # Mismatched profile shapes
        bad = {
            "tree_selection": {
                "target": {"type": "absolute", "var": "stems_per_ha", "amount": 10.0},
                "sets": [{
                    "sfunction": _all_trees,
                    "order_var": "height",
                    "target_var": "stems_per_ha",
                    "target_type": "absolute",
                    "target_amount": 10.0,
                    "profile_x": [0.0, 1.0, 2.0],
                    "profile_y": [0.5, 0.5],  # length mismatch
                    "profile_xmode": "relative",
                }]
            },
            "mode": "odds_units",
            "select_from_all": True,
        }
        with self.assertRaises(MetsiException):
            cutting(stand, **bad)

        # Missing mode
        with self.assertRaises(MetsiException):
            cutting(stand, tree_selection=bad["tree_selection"], freq_var="stems_per_ha", select_from_all=True)

        # Missing select_from_all
        with self.assertRaises(MetsiException):
            cutting(stand, tree_selection=bad["tree_selection"], freq_var="stems_per_ha", mode="odds_units")


if __name__ == "__main__":
    unittest.main()
