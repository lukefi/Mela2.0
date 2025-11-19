import unittest
from unittest.mock import patch

import numpy as np

from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.util.select_units import SelectionSet, SelectionTarget
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.domain.forestry_treatments.mark_trees import mark_trees_fn


class MarkTreesTest(unittest.TestCase):
    def make_empty_stand(self, identifier: str = "stand-A", year: int = 2025) -> ForestStand:
        stand = ForestStand()
        stand.identifier = identifier
        stand.year = year
        return stand

    def make_stand_with_trees(self) -> ForestStand:
        """Create a stand with three reference-tree rows and known stems_per_ha."""
        stand = self.make_empty_stand()

        # Three rows with different stem counts; attributes start as "ORIGINAL"/1
        stems = [10.0, 20.0, 30.0]
        for i, s in enumerate(stems, start=1):
            stand.reference_trees.create(
                {
                    "identifier": f"{stand.identifier}-tree-{i}",
                    "stems_per_ha": s,
                    "tree_type": "ORIGINAL",
                    "management_category": 1,
                }
            )

        return stand

    def test_returns_unchanged_for_empty_stand(self):
        """If there are no reference trees, the treatment should be a no-op."""
        stand = self.make_empty_stand()
        start_size = stand.reference_trees.size

        updated, cdata = mark_trees_fn(stand)

        self.assertIs(updated, stand)
        self.assertEqual(start_size, updated.reference_trees.size)
        self.assertEqual([], cdata)

    def test_missing_tree_selection_raises(self):
        """tree_selection is required when there are reference trees."""
        stand = self.make_stand_with_trees()

        with self.assertRaises(MetsiException):
            # No tree_selection -> should fail validation
            mark_trees_fn(
                stand,
                select_from_all=True,
                mode="odds_units",
                attributes={"tree_type": "SPARE", "management_category": 2},
            )

    @patch("lukefi.metsi.domain.forestry_treatments.mark_trees.select_units")
    def test_full_and_partial_marking_and_splitting(self, mock_select_units):
        """
        Verify that:
          - rows with all stems marked get attributes updated in-place
          - rows with partial stems marked are split into two rows
        """

        stand = self.make_stand_with_trees()
        self.assertEqual(3, stand.reference_trees.size)

        # Current stems_per_ha: [10, 20, 30]
        # We simulate selection where:
        #  - row 0:  0 stems marked  -> unchanged
        #  - row 1: 20 stems marked  -> all stems marked -> in-place attribute update
        #  - row 2:  5 stems marked  -> partial         -> split into 25 + 5 stems
        marked_f = np.array([0.0, 20.0, 5.0], dtype=float)
        mock_select_units.return_value = marked_f

        # Minimal valid tree_selection structure; sfunction/profile not actually used
        def s_all(stand_, trees_):
            # This will never be called because select_units is patched,
            # but the signature must be correct.
            return np.ones(trees_.size, dtype=bool)

        tree_selection = {
            "target": SelectionTarget("absolute", "stems_per_ha", 10.0),
            "sets": [
                SelectionSet[ForestStand, ReferenceTrees](
                    s_all,
                    "breast_height_diameter",
                    "stems_per_ha",
                    "relative",
                    1.0,
                    [0.0, 1.0],
                    [0.0, 1.0],
                    "relative",
                )
            ],
        }

        params = {
            "tree_selection": tree_selection,
            "select_from_all": True,
            "mode": "odds_units",
            "attributes": {
                "tree_type": "SPARE",
                "management_category": 2,
            },
        }

        updated, cdata = mark_trees_fn(stand, **params)

        # select_units should be called once with our stand and reference_trees
        mock_select_units.assert_called_once()

        # No collected data from this treatment
        self.assertEqual([], cdata)

        # We started with 3 rows; one was split into two, so we expect 4 rows now.
        self.assertEqual(4, updated.reference_trees.size)

        stems_after = list(updated.reference_trees.stems_per_ha)
        #  row 0 unchanged, row 1 unchanged, row 2 reduced (30 -> 25), row 3 new (5)
        self.assertEqual([10.0, 20.0, 25.0, 5.0], stems_after)

        tree_types = list(updated.reference_trees.tree_type)
        mgmt_cats = list(updated.reference_trees.management_category)

        # Row 0: no stems marked -> attributes unchanged
        self.assertEqual("ORIGINAL", tree_types[0])
        self.assertEqual(1, mgmt_cats[0])

        # Row 1: all stems marked -> attributes updated in-place
        self.assertEqual("SPARE", tree_types[1])
        self.assertEqual(2, mgmt_cats[1])

        # Row 2: partial stems remain unmarked after split
        self.assertEqual("ORIGINAL", tree_types[2])
        self.assertEqual(1, mgmt_cats[2])

        # Row 3: new row for the marked stems -> attributes set
        self.assertEqual("SPARE", tree_types[3])
        self.assertEqual(2, mgmt_cats[3])


if __name__ == "__main__":
    unittest.main()
