""" Tests suites for forestryfunctions.preprocessing.* modules """
import unittest
import numpy as np

from lukefi.metsi.data.vector_model import TreeStrata
from lukefi.metsi.forestry.preprocessing import pre_util


class TestPreprocessingUtils(unittest.TestCase):
    def _create_strata(self, heights, diameters) -> TreeStrata:
        """
        Helper to create a TreeStrata SoA container with just the fields
        needed for supplement_mean_diameter.
        """
        size = len(heights)
        assert size == len(diameters)

        strata = TreeStrata()
        strata.vectorize(
            {
                "identifier": [f"stratum-{i}" for i in range(size)],
                "mean_height": list(heights),
                "mean_diameter": list(diameters),
            }
        )
        return strata

    def test_supplement_mean_diameter_for_missing_or_zero(self):
        """
        Trees with height > 0 and missing / non-positive diameter
        should get diameter = height * DIAMETER_FACTOR.
        This mirrors the spirit of the old AoS unit test.
        """
        heights = [10.0, 10.0, 10.0]
        diameters = [None, 0.0, None]

        strata = self._create_strata(heights, diameters)

        pre_util.supplement_mean_diameter(strata)

        factor = pre_util.DIAMETER_FACTOR

        # 0: height 10, diameter None -> supplemented
        self.assertAlmostEqual(strata.mean_diameter[0], 10.0 * factor)

        # 1: height 10, diameter 0.0 -> supplemented
        self.assertAlmostEqual(strata.mean_diameter[1], 10.0 * factor)

        # 2: height 10, diameter None -> supplemented
        self.assertAlmostEqual(strata.mean_diameter[2], 10.0 * factor)

    def test_supplement_does_not_touch_valid_or_height_zero(self):
        """
        - Positive diameter values must remain unchanged.
        - Rows with height <= 0 must not be supplemented,
          even if diameter is missing or non-positive.
        """
        heights = [10.0, 0.0, 0.0]
        diameters = [5.0, None, 0.0]

        strata = self._create_strata(heights, diameters)

        pre_util.supplement_mean_diameter(strata)

        # 0: valid diameter, height > 0 -> unchanged
        self.assertAlmostEqual(strata.mean_diameter[0], 5.0)

        # 1: height == 0, diameter None -> still NaN (no supplementation)
        self.assertTrue(np.isnan(strata.mean_diameter[1]))

        # 2: height == 0, diameter 0.0 -> still 0.0 (no supplementation)
        self.assertEqual(strata.mean_diameter[2], 0.0)
