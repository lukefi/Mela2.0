import unittest

from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.forestry_treatments.soil_surface_preparation import soil_surface_preparation

class SoilSurfacePreparationTest(unittest.TestCase):
    def test_sets_soil_surface_preparation_year_and_no_cdata(self):
        stand = ForestStand()
        stand.identifier = "stand-SSP"
        stand.year = 2030

        updated, cdata = soil_surface_preparation(stand)
        self.assertEqual([], cdata)
        self.assertEqual(2030, updated.soil_surface_preparation_year)


if __name__ == "__main__":
    unittest.main()
