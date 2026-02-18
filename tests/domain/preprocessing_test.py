import unittest
import lukefi.metsi.domain.pre_ops as preprocessing
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.enums.internal import TreeSpecies
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.forestry.preprocessing.coordinate_conversion import CRS
from lukefi.metsi.app.utils import MetsiException


class PreprocessingTest(unittest.TestCase):

    def test_generate_reference_trees(self):
        # Set up a single TreeStratum in SoA form
        stand = ForestStand()
        stand.identifier = "xxx"
        stand.area = 200.0
        stand.area_weight_factors = (1.0, 0.6)

        stand.tree_strata = TreeStrata().vectorize(
            {
                "identifier": ["1-stratum"],
                "species": [TreeSpecies.PINE.value],
                "mean_diameter": [17.0],
                "mean_height": [15.0],
                "basal_area": [250.0],
                "stems_per_ha": [None],
                "biological_age": [10.0],
            }
        )

        # No measured trees initially
        stand.reference_trees = ReferenceTrees()

        result = preprocessing.generate_reference_trees([stand], n_trees=10)
        result_stand = result[0]
        trees = result_stand.reference_trees

        # SoA: use size and arrays instead of *_pre_vec
        self.assertEqual(10, trees.size)
        self.assertEqual("xxx-1-tree", trees.identifier[0])
        self.assertEqual(10237.96, trees.stems_per_ha[0])
        self.assertEqual(1138.02, trees.stems_per_ha[1])

        self.assertEqual(0.0, result_stand.area_weight)

    def test_scale_area_weight(self):
        stand = ForestStand(area_weight=100.0, area_weight_factors=(0.0, 1.2))
        result = preprocessing.scale_area_weight([stand])
        self.assertEqual(result[0].area_weight, 120.0)

    def test_coordinate_conversion_operation(self):
        dummy_float = 0.0
        crs = CRS.EPSG_3067.name
        stand = ForestStand(
            geo_location=(6640610.26, 267924.92, dummy_float, crs)
        )
        one_stand_list = [stand]
        valid_assertions = [
            # these are the valid config level inputs
            {},  # empty, default
            {"target_system": "YKJ"},
            {"target_system": "EPSG:2393"},
        ]
        for asse in valid_assertions:
            result = preprocessing.convert_coordinates(one_stand_list, **asse)
            rstand = result[0]
            if rstand.geo_location:
                self.assertEqual(rstand.geo_location[0], 6643400.000631507)
                self.assertEqual(rstand.geo_location[1], 3268000.003019635)
                self.assertEqual(rstand.geo_location[3], CRS.EPSG_2393.name)
            invalid_assertion = {"target_system": "ASD"}
            self.assertRaises(
                Exception,
                preprocessing.convert_coordinates,
                [ForestStand()],
                **invalid_assertion,
            )

    def test_compute_location_metadata(self):
        # generate testing data
        latitude = 6643400.000631507
        longitude = 3268000.003019635
        sea_level_heights = [25.0, 25.0, None]
        valid_crs = ["EPSG:3067", "EPSG:2393", "EPSG:2393"]
        valid_fixtures = [
            ForestStand(geo_location=(latitude, longitude, sl_height, crs))
            for crs, sl_height in zip(valid_crs, sea_level_heights)
        ]

        assertions = [
            (
                sea_level_heights[0],
                valid_crs[0],
                1674,
                0.0,
                -26.58841390118755,
                -25.583422976421808,
                52.32,
                40.22,
            ),
            (
                sea_level_heights[1],
                valid_crs[1],
                1674,
                0.0,
                -4.41760238263144,
                -4.988233118838529,
                52.287995264010235,
                36.23342105952951,
            ),
            (
                1.6666666666666667,
                valid_crs[2],
                1674,
                0.0,
                -4.319369049298107,
                -4.8948997855051966,
                52.287995264010235,
                36.23342105952951,
            ),
        ]

        results = preprocessing.compute_location_metadata(valid_fixtures)

        for result, asse in zip(results, assertions):
            geo = result.geo_location
            if geo:
                self.assertIsNotNone(geo)
                self.assertIsNotNone(geo[2])

                # actual test validation
                self.assertEqual(geo[0], latitude)
                self.assertEqual(geo[1], longitude)
                if geo[2]:
                    self.assertEqual(float(geo[2]), asse[0])
                if geo[3]:
                    self.assertEqual(geo[3], asse[1])
            self.assertEqual(results[0].degree_days, asse[2])
            self.assertEqual(results[0].sea_effect, asse[3])

        invalid_fixtures = [
            [ForestStand(geo_location=(None, None, None, None))],
            [ForestStand(geo_location=(1, None, None, None))],
            [ForestStand(geo_location=(None, 1, None, None))],
            [ForestStand(geo_location=(1, 1, 1, "DUMMY_CRS"))],
        ]
        for invalid in invalid_fixtures:
            # Exception testing
            self.assertRaises(
                MetsiException, preprocessing.compute_location_metadata, invalid
            )
