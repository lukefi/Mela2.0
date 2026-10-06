import unittest
import numpy as np
import numpy.testing as nptest

from parameterized import parameterized

from lukefi.metsi.data.enums.internal import Storey, TreeManagementCategory, TreeSpecies
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.domain.pre_ops import supplement_storey_information
from lukefi.metsi.forestry.storey import (
    calc_storey_basal_area,
    calc_storey_dominant_height,
    calc_tree_basal_areas,
    stand_has_only_retention_storey,
    stand_has_only_seeding_tree_storey)


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

    @parameterized.expand([
        ([10.0] * 6,
         [10.0, 40.0, 30.0, 2.0, 14.0, 2.0],
         [5.0] * 6,
         [
            Storey.DOMINANT,
            Storey.OVER,
            Storey.OVER,
            Storey.UNDER,
            Storey.DOMINANT,
            Storey.UNDER
        ]),
        ([1000.0] * 6,
         [10.0, 40.0, 30.0, 2.0, 54.0, 46.0],
         [9.0, 15.0, 5.0, 4.0, 12.0, 18.0],
         [
            Storey.UNDER,
            Storey.DOMINANT,
            Storey.UNDER,
            Storey.UNDER,
            Storey.OVER,
            Storey.OVER
        ]),
        ([1000.0] * 6,
         [38.0, 40.0, 30.0, 2.0, 54.0, 46.0],
         [0.0] * 6,
         [
            Storey.DOMINANT,
            Storey.DOMINANT,
            Storey.UNDER,
            Storey.UNDER,
            Storey.OVER,
            Storey.OVER
        ]
        )
    ])
    def test_supplement_storey_information(self, stems_per_ha, height, breast_height_diameter, expected):
        trees = ReferenceTrees()
        trees.vectorize(
            {
                "identifier": [f"tree{i}" for i in range(1, 7)],
                "storey": [
                    Storey.UNDER,
                    Storey.OVER,
                    Storey.INDETERMINATE,
                    Storey.UNSET,
                    Storey.REMOVAL,
                    Storey.REMOTE],
                "stems_per_ha": stems_per_ha,
                "height": height,
                "species": [TreeSpecies.PINE] * 6,
                "breast_height_diameter": breast_height_diameter
            }
        )
        stand = ForestStand(trees, identifier="test")
        stand = supplement_storey_information([stand])[0]

        self.assertTrue(np.all(trees.storey == np.asarray(expected)))


class StoreyUtilitiesTest(unittest.TestCase):

    def test_stand_has_only_retention_storey(self):
        trees1 = ReferenceTrees(3)
        trees1.storey[:] = Storey.RETENTION

        trees2 = ReferenceTrees(3)
        trees2.storey[0] = Storey.DOMINANT
        trees2.storey[1:] = Storey.RETENTION

        trees3 = ReferenceTrees(3)
        trees3.storey[0] = Storey.UNDER
        trees3.storey[1] = Storey.OVER
        trees3.storey[2] = Storey.REMOTE

        self.assertTrue(stand_has_only_retention_storey(trees1))
        self.assertFalse(stand_has_only_retention_storey(trees2))
        self.assertFalse(stand_has_only_retention_storey(trees3))

    def test_stand_has_only_seeding_tree_storey(self):
        trees1 = ReferenceTrees(3)
        trees1.storey[:] = Storey.OVER
        trees1.management_category[:] = TreeManagementCategory.SEEDING_TREE

        trees2 = ReferenceTrees(3)
        trees2.storey[:] = Storey.OVER
        trees2.management_category[0] = TreeManagementCategory.NO_RESTRICTION
        trees2.management_category[1:] = TreeManagementCategory.SEEDING_TREE

        trees3 = ReferenceTrees(3)
        trees3.storey[0] = Storey.UNDER
        trees3.storey[1:] = Storey.OVER
        trees3.management_category[:] = TreeManagementCategory.SEEDING_TREE

        self.assertTrue(stand_has_only_seeding_tree_storey(trees1))
        self.assertFalse(stand_has_only_seeding_tree_storey(trees2))
        self.assertFalse(stand_has_only_seeding_tree_storey(trees3))

    def test_calc_tree_basal_areas(self):
        diameters = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64)
        bas = calc_tree_basal_areas(diameters)

        nptest.assert_allclose(bas, [7.85398163e-05, 3.14159265e-04, 7.06858347e-04, 1.25663706e-03,
                                     1.96349541e-03, 2.82743339e-03, 3.84845100e-03, 5.02654825e-03])

    def test_calc_storey_basal_area(self):
        bas = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        stems = np.asarray([5.0, 4.0, 3.0, 2.0], dtype=np.float64)

        ba = calc_storey_basal_area(bas, stems)

        self.assertAlmostEqual(30.0, ba)

    def test_calc_storey_dominant_height_one_tree_dominates(self):
        trees = ReferenceTrees()
        trees.vectorize(
            {
                "identifier": [f"tree{i}" for i in range(10)],
                "stems_per_ha": [600, 600, 200, 200, 100, 100, 100, 100, 100, 100],
                "breast_height_diameter": [10, 5, 20, 30, 10, 10, 5, 5, 5, 15],
                "height": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "storey": [Storey.DOMINANT] * 10
            }
        )

        dh = calc_storey_dominant_height(trees, Storey.DOMINANT)

        assert dh is not None

        self.assertAlmostEqual(4.0, dh)

    def test_calc_storey_dominant_height_less_than_100_stems(self):
        trees = ReferenceTrees()
        trees.vectorize(
            {
                "stems_per_ha": [10.0, 20.0, 30.0],
                "breast_height_diameter": [5, 6, 4],
                "height": [15, 16, 17],
                "storey": [Storey.DOMINANT] * 3
            }
        )

        dh = calc_storey_dominant_height(trees, Storey.DOMINANT)

        assert dh is not None

        self.assertAlmostEqual(dh, 16.333333333333332)

    def test_calc_storey_dominant_height_standard_case(self):
        trees = ReferenceTrees()
        trees.vectorize(
            {
                "stems_per_ha": [10.0, 30.0, 5.0, 60.0, 40.0],
                "breast_height_diameter": [17.0, 2.0, 4.0, 9.0, 7.0],
                "height": [20.0, 14.0, 14.0, 12.0, 15.0],
                "storey": [Storey.DOMINANT] * 5
            }
        )

        dh = calc_storey_dominant_height(trees, Storey.DOMINANT)

        assert dh is not None

        self.assertAlmostEqual(dh, 13.7)
