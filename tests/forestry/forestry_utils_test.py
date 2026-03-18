import unittest

from lukefi.metsi.forestry import forestry_utils as futil
from lukefi.metsi.data.enums.internal import TreeSpecies, Storey
from lukefi.metsi.data.vector_model import ReferenceTree, TreeStrata, TreeStratum


def strata_fixture() -> list[TreeStratum]:
    values = [
        {
            "species": TreeSpecies.SPRUCE,
            "storey": Storey.DOMINANT,
            "mean_diameter": 10.0,
        },
        {
            "species": TreeSpecies.SPRUCE,
            "storey": Storey.UNDER,
            "mean_diameter": 5.0,
        },
        {
            "species": TreeSpecies.PINE,
            "storey": Storey.DOMINANT,
            "mean_diameter": 18.0,
        },
        {
            "species": TreeSpecies.DOWNY_BIRCH,
            "storey": Storey.UNDER,
            "mean_diameter": 5.0,
        },
        {
            "species": TreeSpecies.SPRUCE,
            "storey": Storey.DOMINANT,
            "mean_diameter": 5.0,
        },
        {
            "species": TreeSpecies.COMMON_ALDER,
            "storey": Storey.UNDER,
            "mean_diameter": 6.2,
        },
        {
            "species": TreeSpecies.COMMON_ASH,
            "storey": Storey.UNDER,
            "mean_diameter": 6.5,
        },
    ]
    return [TreeStratum(**v) for v in values]


class ForestryUtilsTest(unittest.TestCase):
    #
    # AoS-based tests
    #
    def test_calculate_basal_area(self):
        tree = ReferenceTree()
        assertions = [
            ((10.0, 50.0), 0.3927),
            ((0.0, 50.0), 0.0),
            ((10.0, 0.0), 0.0),
            ((0.0, 0.0), 0.0),
        ]
        for (diameter, stems), expected in assertions:
            tree.breast_height_diameter = diameter
            tree.stems_per_ha = stems
            result = futil.calculate_basal_area(tree)
            self.assertEqual(expected, round(result, 4))

    def test_generate_diameter_threshold(self):
        assertions = [
            ((10.0, 20.0), 13.33333),
            ((5.0, 10.0), 6.66667),
            ((0.0, 10.0), 0.0),
            ((5.0, 0.0), 0.0),
        ]

        for (d1, d2), expected in assertions:
            result = round(
                futil.generate_diameter_threshold(d1, d2),
                5,
            )
            self.assertEqual(result, expected)

    def test_override_from_diameter(self):
        initial_stratum = TreeStratum()
        initial_stratum.mean_diameter = 10.0
        current_stratum = TreeStratum()
        current_stratum.mean_diameter = 20.0
        assertions = [
            (13.0, current_stratum),
            (15.0, initial_stratum),
        ]
        for diameter, expected_stratum in assertions:
            reference_tree = ReferenceTree()
            reference_tree.breast_height_diameter = diameter
            result = futil.override_from_diameter(
                initial_stratum, current_stratum, reference_tree
            )
            self.assertEqual(expected_stratum, result)

    #
    # SoA-based tests for find_matching_storey_stratum_for_tree
    #
    def test_matching_storey_stratum_same_species(self):
        """
        Simple SoA case: same species + same storey, pick closest mean_diameter.
        """
        strata = TreeStrata().vectorize(
            {
                "identifier": ["s1", "s2"],
                "species": [
                    TreeSpecies.SPRUCE.value,
                    TreeSpecies.SPRUCE.value,
                ],
                "storey": [
                    Storey.DOMINANT.value,
                    Storey.DOMINANT.value,
                ],
                "mean_diameter": [10.0, 20.0],
                "asema": ["1", "1"]
            }
        )
        tree = ReferenceTree(
            identifier="t1",
            species=TreeSpecies.SPRUCE,
            storey=Storey.DOMINANT,
            breast_height_diameter=11.0,
            crown_class="2"
        )

        stratum = futil.find_matching_storey_stratum_for_tree(tree, strata)
        # 11 cm is closer to 10 cm than 20 cm => index 0
        self.assertIsNotNone(stratum)
        self.assertEqual("s1", stratum.identifier)

    def test_matching_storey_stratum_similar_deciduous_species(self):
        """
        SoA case: deciduous tree, UNDER storey, should use "similar species"
        logic and pick closest deciduous stratum in same storey.
        """
        strata = TreeStrata().vectorize(
            {
                "identifier": ["s1", "s2"],
                # Two UNDER-storey deciduous strata, different species
                "species": [
                    TreeSpecies.SILVER_BIRCH.value,   # similar birch
                    TreeSpecies.COMMON_ALDER.value,   # other deciduous
                ],
                "storey": [
                    Storey.UNDER.value,
                    Storey.UNDER.value,
                ],
                "mean_diameter": [6.0, 10.0],
                "asema": [5, 5, 5]
            }
        )
        tree = ReferenceTree(
            identifier="t1",
            species=TreeSpecies.DOWNY_BIRCH,
            storey=Storey.UNDER,
            breast_height_diameter=6.2,
            crown_class="5"
        )

        stratum = futil.find_matching_storey_stratum_for_tree(tree, strata)
        # 6.2 cm is closer to mean_diameter 6.0 than 10.0 => index 0
        self.assertEqual("s1", stratum.identifier)

    def test_matching_storey_stratum_invalid_diameter(self):
        """
        SoA case: diameter 0 or NaN gives no match.
        """
        # Diameter 0.0
        strata = TreeStrata().vectorize(
            {
                "identifier": ["s1"],
                "species": [TreeSpecies.SPRUCE.value],
                "storey": [Storey.DOMINANT.value],
                "mean_diameter": [10.0],
                "asema": [1]
            }
        )
        tree_zero = ReferenceTree(
            identifier="t1",
            species=TreeSpecies.SPRUCE,
            storey=Storey.DOMINANT,
            breast_height_diameter=0.0,
            crown_class="2"
        )
        idx_zero = futil.find_matching_storey_stratum_for_tree(tree_zero, strata)
        self.assertIsNone(idx_zero)

        tree_nan = ReferenceTree(
            identifier="t1",
            species=TreeSpecies.SPRUCE,
            storey=Storey.DOMINANT,
            crown_class="2"
        )
        idx_nan = futil.find_matching_storey_stratum_for_tree(tree_nan, strata)
        self.assertIsNone(idx_nan)

    def test_matching_storey_stratum_threshold_respected(self):
        """
        SoA case: using a stricter diameter_threshold should cause an
        otherwise-closest stratum to be rejected.
        """
        strata = TreeStrata().vectorize(
            {
                "identifier": ["s1", "s2"],
                "species": [
                    TreeSpecies.SPRUCE.value,
                    TreeSpecies.SPRUCE.value,
                ],
                "storey": [
                    Storey.DOMINANT.value,
                    Storey.DOMINANT.value,
                ],
                "mean_diameter": [10.0, 20.0],
                "asema": ["1", "1"]
            }
        )
        tree = ReferenceTree(
            identifier="t1",
            species=TreeSpecies.SPRUCE,
            storey=Storey.DOMINANT,
            breast_height_diameter=40.0,
            crown_class="2"
        )

        # With default threshold=2.5, 40 is within [20/2.5, 20*2.5] = [8, 50] → match index 1
        stratum_default = futil.find_matching_storey_stratum_for_tree(tree, strata)
        self.assertEqual("s2", stratum_default.identifier)

        # With stricter threshold, 40 is outside allowed range -> None
        stratum_strict = futil.find_matching_storey_stratum_for_tree(tree, strata, diameter_threshold=1.5)
        self.assertIsNone(stratum_strict)
