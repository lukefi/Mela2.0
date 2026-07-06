import unittest
from lukefi.metsi.forestry.preprocessing.coordinate_conversion import (
    convert_location_to_ykj, erts_tm35_to_ykj)
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.exceptions import MetsiException


class TestCoordinateConversion(unittest.TestCase):
    def test_coordinate_conversion(self):
        u = 6640610.26
        v = 267924.92
        (x, y) = erts_tm35_to_ykj(u, v)
        self.assertEqual(x, 6643400.000631507)
        self.assertEqual(y, 3268000.003019635)

    def test_convert_location_to_ykj(self):
        dummy_float = 0.0
        target_crs = 'EPSG:2393'
        passthrough_gl: tuple[float, float, float, str] = (dummy_float, dummy_float, dummy_float, target_crs)

        stand_assertion = ForestStand(geo_location=passthrough_gl)
        assert stand_assertion.geo_location is not None

        result = convert_location_to_ykj(*passthrough_gl)
        self.assertEqual(result[0], dummy_float)
        self.assertEqual(result[1], dummy_float)
        self.assertEqual(result[3], target_crs)
        # Valid for YKJ-conversion
        valid_gl = (6640610.26, 267924.92, dummy_float, 'EPSG:3067')
        stand_assertion.geo_location = valid_gl
        result = convert_location_to_ykj(*stand_assertion.geo_location)
        self.assertEqual(result[0], 6643400.000631507)
        self.assertEqual(result[1], 3268000.003019635)
        self.assertEqual(result[3], target_crs)
        # Invalid CRS raises exception
        invalid_gl: tuple[float, float, float, str] = (dummy_float, dummy_float, dummy_float, "InvalidCRS")

        self.assertRaises(MetsiException, convert_location_to_ykj, *invalid_gl)
