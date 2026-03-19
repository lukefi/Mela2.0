import unittest

import numpy as np

from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import TreeStrata, ReferenceTrees


class TestForestDataModel(unittest.TestCase):
    def test_tree_strata_get_stratum(self):
        strata = TreeStrata()
        strata.create([{
            "species": 1,
            "mean_height": 2.0,
            "mean_diameter": 1.0,
            "breast_height_age": 10.0,
            "biological_age": 12.0,
            "sapling_stems_per_ha": 1200.0,
        }])

        row = strata.get_stratum(0)

        self.assertEqual(1, int(row.species))
        self.assertEqual(2.0, row.mean_height)
        self.assertEqual(1.0, row.mean_diameter)
        self.assertEqual(10.0, row.breast_height_age)
        self.assertEqual(12.0, row.biological_age)
        self.assertEqual(1200.0, row.sapling_stems_per_ha)

    def test_reference_trees_get_tree(self):
        trees = ReferenceTrees()
        trees.create([{
            "species": 1,
            "breast_height_diameter": 11.5,
            "biological_age": 10.0,
            "stems_per_ha": 100.0,
        }])

        tree = trees.get_tree(0)

        self.assertEqual(11.5, tree.breast_height_diameter)
        self.assertEqual(10.0, tree.biological_age)
        self.assertEqual(100.0, tree.stems_per_ha)

    def test_set_area_without_weight(self):
        fixture = ForestStand()
        fixture.set_area(1.0)
        self.assertEqual(1.0, fixture.area)

    def test_set_area_with_weight(self):
        fixture = ForestStand()
        fixture.set_area(1.0)
        self.assertEqual(1.0, fixture.area)
        self.assertEqual(1.0, fixture.area_weight)

    def test_set_geo_location(self):
        fixture = ForestStand()
        assertions = [
            ((6000.1, 304.3, 10.0), (6000.1, 304.3, 10.0, 'EPSG:3067')),
            ((6000.1, 304.3, None), (6000.1, 304.3, None, 'EPSG:3067'))
        ]
        failures = [
            (None, 20.3, 20),
            (23.4, None, 20),

        ]
        for i in assertions:
            fixture.set_geo_location(*i[0])
            self.assertEqual(i[1], fixture.geo_location)
        for i in failures:
            self.assertRaises(Exception, lambda: fixture.set_geo_location(*i))

    def test_convert_csv_stand_row_with_missing_altitude(self):
        row = "stand;12345;2018;436.0;436.0;6834156.23;429291.91;None;EPSG:3067;1019.0;" \
              "4;1;2;" \
              "3;0;3;8;1984;None;2018;None;0;None;None;" \
              "None;10;1;None;12;1;0;False;1.0;1.0;1;10;51;None;None;1;1;0;0"
        row = row.split(';')
        stand = ForestStand.from_csv_row(row)

        self.assertEqual((6834156.23, 429291.91, None, 'EPSG:3067'), stand.geo_location)

    def test_update_aggregates(self):
        stand = ForestStand()
        stand.reference_trees.create([{"breast_height_diameter": 12, "stems_per_ha": 5, "height": 13, "species": 1},
                                     {"breast_height_diameter": 5, "stems_per_ha": 14, "height": 10, "species": 3}])
        stand.tree_strata.create([{"stems_per_ha": 39.2, "basal_area": 23.4,
                                 "mean_diameter": 14.9, "mean_height": 5.9},
                                  {"stems_per_ha": 0.6, "basal_area": 0.12,
                                   "mean_diameter": 1.3, "mean_height": 2.2}])
        stand.update_aggregates()

        self.assertTrue(
            np.all(np.isclose(np.asarray([0.01130973355, 0.00196349540]), stand.reference_trees.basal_area)))

        self.assertAlmostEqual(58.8, stand.stems_per_ha or 0)
        self.assertAlmostEqual(23.60403760335, stand.basal_area or 0)
        self.assertAlmostEqual(14.81238229506, stand.ds_ba_weighted_mean_diameter or 0)
        self.assertAlmostEqual(5.902974074951, stand.ds_ba_weighted_mean_height or 0)
