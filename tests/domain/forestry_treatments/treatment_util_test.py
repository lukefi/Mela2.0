import unittest

import numpy as np

from lukefi.metsi.domain.forestry_treatments.treatment_util import collect_removed_trees
from lukefi.metsi.data.vector_model import ReferenceTrees

class TreatmentUtilTest(unittest.TestCase):
    def test_collect_removed_trees(self):

        # Original ReferenceTrees
        rts_original = ReferenceTrees(size=4)
        rts_original.tree_number = np.array([1, 2, 3, 4], dtype=int)
        rts_original.stems_per_ha = np.array([100.0, 200.0, 300.0, 400.0])
        rts_original.species = np.array([1, 1, 2, 2], dtype=int)
        rts_original.origin = np.array([1, 1, 2, 2], dtype=int)
        rts_original.height = np.array([10.0, 20.0, 30.0, 40.0])
        rts_original.biological_age = np.array([5.0, 10.0, 15.0, 20.0])

        # Modified ReferenceTrees (tree 2 reduced and tree 3 removed)
        rts_modified = ReferenceTrees(size=3)
        rts_modified.tree_number = np.array([1, 2, 4], dtype=int)
        rts_modified.stems_per_ha = np.array([100.0, 150.0, 400.0])  # tree 2 reduced from 200 to 150
        rts_modified.species = np.array([1, 1, 2], dtype=int)
        rts_modified.origin = np.array([1, 1, 2], dtype=int)
        rts_modified.height = np.array([10.0, 20.0, 40.0])
        rts_modified.biological_age = np.array([5.0, 10.0, 20.0])

        removed_trees = collect_removed_trees(rts_original, rts_modified)

        # Expected removed trees: tree_number=2 (reduced) and tree_number=3 (removed)
        expected_tree_numbers = [2, 3]
        expected_stems_per_ha = [50.0, 300.0]  # tree_number=2 reduced by (200-150)=50 and tree_number=3 removed completely

        self.assertEqual(list(removed_trees.tree_number), expected_tree_numbers)
        self.assertEqual(list(removed_trees.stems_per_ha), expected_stems_per_ha)