from collections import namedtuple
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import TreeStrata
from lukefi.metsi.forestry.preprocessing import tree_generation

from tests.forestry import test_util


class TestDistributions(test_util.ConverterTestSuite):
    Input = namedtuple(
        "Input",
        "species diameter basal_area mean_height breast_height_age biological_age stand stems_per_ha sapling_stems_per_ha")

    def create_test_stratums(self, inputs):

        strata = TreeStrata()
        strata.create([
            {
                "species": i.species,
                "mean_diameter": i.diameter,
                "basal_area": i.basal_area,
                "mean_height": i.mean_height,
                "breast_height_age": i.breast_height_age,
                "biological_age": i.biological_age,
                "stems_per_ha": i.stems_per_ha,
                "sapling_stems_per_ha": i.sapling_stems_per_ha,
            }
            for i in inputs
        ])
        return [strata.get_stratum(i) for i in range(strata.size)]

    def test_generate_reference_trees_from_height_distribution(self):
        # Test data generation
        n_trees = 10
        stand = ForestStand()
        stand.identifier = "0-023-002-02-1"
        stratum_inputs = [
            # species=1;
            # mean_diameter=1.8;
            # basal_area=0.0;
            # mean_height=2.5;
            # breast_height_age=8;
            # biological_age=12;
            # ForestStand=stand;
            # stems_per_ha=None
            # sapling_stems_per_ha=330.0
            self.Input(1, 1.8, 0.0, 2.5, 8, 12, stand, 330.0, 0.0),  # Sapling tree generation
        ]
        test_data = self.create_test_stratums(stratum_inputs)

        expected_results = [
            # n_trees=1;
            # ForestStand=stand;
            # species=1;
            # diameter=1.78;
            # height=2.19;
            # stems_per_ha=33.0;
            # breast_height_age=0.0;
            # biological_age=5;
            [1, stand, 0.0, 1.57, 2.04, 33.0, 8, 12]
        ]

        # Derive results
        results = []
        for test_stratum in test_data:
            result = tree_generation._trees_from_sapling_height_distribution(test_stratum, n_trees=n_trees)
            results.append(result)
        # Validate
        for (result, asse) in zip(results, expected_results):
            self.assertEqual(asse[3], round(result[0].breast_height_diameter, 2))
            self.assertEqual(asse[4], round(result[0].height, 2))
            self.assertEqual(asse[5], round(result[0].stems_per_ha, 2))
