import unittest
import numpy as np
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata


class TestVectorize(unittest.TestCase):

    def test_reference_trees_vectorize_defaults_and_types(self):
        vec = ReferenceTrees().vectorize({
            "identifier": ["t-1", "t-2"],
            "tree_number": [1, 2],
            "species": [1, 2],
            # intentionally leave other fields missing -> defaults
        })

        self.assertEqual(len(vec), 2)
        self.assertIsInstance(vec.species, np.ndarray)
        self.assertEqual(vec.species.dtype, np.int32)
        self.assertListEqual(vec.identifier.tolist(), ["t-1", "t-2"])

        # Missing float fields default to NaN
        self.assertTrue(np.isnan(vec.height[0]))
        # Missing ints default to -1
        self.assertEqual(int(vec.origin[0]), -1)
        # Missing bools default to False
        self.assertFalse(bool(vec.sapling[0]))
        # Missing strings default to empty string
        self.assertEqual(str(vec.tree_type[0]), "")

    def test_tree_strata_vectorize_and_slice(self):
        strata = TreeStrata().vectorize({
            "identifier": ["s-1", "s-2", "s-3"],
            "species": [1, 2, 3],
            "stems_per_ha": [100.0, 200.0, 300.0],
        })

        self.assertEqual(strata.size, 3)

        sub = strata[1:]
        self.assertIsInstance(sub, TreeStrata)
        self.assertEqual(sub.size, 2)
        self.assertListEqual(sub.identifier.tolist(), ["s-2", "s-3"])
