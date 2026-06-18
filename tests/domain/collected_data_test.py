from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.domain.collected_data import NaturalProcessInfo
import unittest


class NaturalProcessInfoTest(unittest.TestCase):
    before: ReferenceTrees
    after: ReferenceTrees
    npi: NaturalProcessInfo

    @classmethod
    def setUpClass(cls) -> None:
        cls.before = ReferenceTrees()
        cls.before.vectorize(
            {
                "identifier": ["tree1", "tree2"],
                "stratum": [1, 2],
                "stems_per_ha": [1.0, 2.0],
                "breast_height_diameter": [1.0, 1.5],
                "height": [5.0, 5.0]
            }
        )

        # tree2 has been split into two trees, tree3 and tree4
        cls.after = ReferenceTrees()
        cls.after.vectorize(
            {
                "identifier": ["tree1", "tree3", "tree4"],
                "stratum": [1, 2, 2],
                "stems_per_ha": [1.5, 1.2, 1.3],
                "breast_height_diameter": [1.1, 1.6, 1.7],
                "height": [5.5, 5.6, 5.7]
            }
        )
        cls.npi = NaturalProcessInfo()
        cls.npi.start_year = 2020
        cls.npi.step = 5
        cls.npi.trees_before = cls.before
        cls.npi.trees_after = cls.after


    def test_get_stems_per_ha_after(self):
        self.assertEqual(1.5, self.npi._get_stems_per_ha_after(0))
        self.assertEqual(2.5, self.npi._get_stems_per_ha_after(1))

    def test_get_breast_height_diameter_after(self):
        self.assertEqual(1.1, self.npi._get_breast_height_diameter_after(0))
        self.assertAlmostEqual(1.6550153756040415, self.npi._get_breast_height_diameter_after(1))

    def test_get_height_after(self):
        self.assertEqual(5.5, self.npi._get_height_after(0))
        self.assertAlmostEqual(5.655015375604042, self.npi._get_height_after(1))
