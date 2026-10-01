import unittest
import numpy as np

from lukefi.metsi.data.enums.internal import Storey, TreeManagementCategory, TreeSpecies
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.domain.pre_ops import supplement_storey_information


class StoreySupplementingTest(unittest.TestCase):

    def test_promote_unset_to_dominant(self):
        trees = ReferenceTrees()
        trees.vectorize(
            {
                "identifier": [f"tree{i}" for i in range(1, 11)],
                "storey": [Storey.UNSET] * 10
            }
        )
        stand = ForestStand(trees, identifier="test")
        stand = supplement_storey_information([stand])[0]

        self.assertTrue(np.all(trees.storey == Storey.DOMINANT))

    def test_promote_indeterminate_to_dominant(self):
        trees = ReferenceTrees()
        trees.vectorize(
            {
                "identifier": [f"tree{i}" for i in range(1, 11)],
                "storey": [Storey.INDETERMINATE] * 10
            }
        )
        stand = ForestStand(trees, identifier="test")
        stand = supplement_storey_information([stand])[0]

        self.assertTrue(np.all(trees.storey == Storey.DOMINANT))

    def test_promote_under_to_dominant(self):
        trees = ReferenceTrees()
        trees.vectorize(
            {
                "identifier": [f"tree{i}" for i in range(1, 11)],
                "storey": [Storey.UNDER] * 10
            }
        )
        stand = ForestStand(trees, identifier="test")
        stand = supplement_storey_information([stand])[0]

        self.assertTrue(np.all(trees.storey == Storey.DOMINANT))

    def test_promote_non_seeding_over_to_dominant(self):
        trees = ReferenceTrees()
        trees.vectorize({"identifier": [f"tree{i}" for i in range(1, 11)],
                        "storey": [Storey.OVER] * 10,
                         "management_category": [TreeManagementCategory.NO_RESTRICTION] * 5 +
                         [TreeManagementCategory.SEEDING_TREE] * 5,
                         "stems_per_ha": [1.0] * 10, "height": [20.0] * 5 + [30.0] * 5
                         })
        stand = ForestStand(trees, identifier="test")
        stand = supplement_storey_information([stand])[0]

        self.assertTrue(np.all(trees.storey == np.asarray([Storey.DOMINANT] * 5 + [Storey.OVER] * 5, dtype=np.int32)))

    def test_promote_remote_to_dominant(self):
        trees = ReferenceTrees()
        trees.vectorize(
            {
                "identifier": [f"tree{i}" for i in range(1, 11)],
                "storey": [Storey.REMOTE] * 10
            }
        )
        stand = ForestStand(trees, identifier="test")
        stand = supplement_storey_information([stand])[0]

        self.assertTrue(np.all(trees.storey == Storey.DOMINANT))

    def test_promote_removal_to_dominant(self):
        trees = ReferenceTrees()
        trees.vectorize(
            {
                "identifier": [f"tree{i}" for i in range(1, 11)],
                "storey": [Storey.REMOVAL] * 10
            }
        )
        stand = ForestStand(trees, identifier="test")
        stand = supplement_storey_information([stand])[0]

        self.assertTrue(np.all(trees.storey == Storey.DOMINANT))

    def test_dont_promote_retention_to_dominant(self):
        trees = ReferenceTrees()
        trees.vectorize(
            {
                "identifier": [f"tree{i}" for i in range(1, 11)],
                "storey": [Storey.RETENTION] * 10
            }
        )
        stand = ForestStand(trees, identifier="test")
        stand = supplement_storey_information([stand])[0]

        self.assertTrue(np.all(trees.storey == Storey.RETENTION))

    def test_promote_dominant_from_many(self):
        trees = ReferenceTrees()
        trees.vectorize(
            {
                "identifier": [f"tree{i}" for i in range(1, 7)],
                "storey": [Storey.UNDER,
                           Storey.OVER,
                           Storey.INDETERMINATE,
                           Storey.UNSET,
                           Storey.REMOVAL,
                           Storey.REMOTE],
                "stems_per_ha": [10.0] * 6,
                "height": [10.0, 40.0, 30.0, 2.0, 14.0, 2.0],
                "species": [TreeSpecies.PINE] * 6,
                "breast_height_diameter": [5.0] * 6
            }
        )
        stand = ForestStand(trees, identifier="test")
        stand = supplement_storey_information([stand])[0]

        self.assertTrue(np.all(trees.storey == np.asarray(
            [
                Storey.DOMINANT,
                Storey.OVER,
                Storey.OVER,
                Storey.UNDER,
                Storey.DOMINANT,
                Storey.UNDER
            ])))
