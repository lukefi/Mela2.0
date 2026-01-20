import unittest
from unittest.mock import Mock
from lukefi.metsi.data.vector_model import ReferenceTrees

from lukefi.metsi.data.model import ReferenceTree, ForestStand, TreeStratum
from lukefi.metsi.domain.forestry_types import StandList
from lukefi.metsi.app.utils import ConfigurationException
from lukefi.metsi.domain.exp_ops import prepare_rst_output, classify_values_to
from lukefi.metsi.data.enums.internal import TreeSpecies


class TestExpOps(unittest.TestCase):

    def _make_reference_trees(self, categories: list[str]) -> ReferenceTrees:
        """
        Helper: create a ReferenceTrees SoA container with given tree_category codes.
        First entry is living (e.g. ""), others can be dead (e.g. "2").
        """
        vec = ReferenceTrees()
        n = len(categories)

        attr = {
            "identifier": [f"t{i + 1}" for i in range(n)],
            "tree_number": [i + 1 for i in range(n)],
            "tree_category": categories,
        }
        vec.vectorize(attr)
        return vec

    def setUp(self):
        # stand1: forest land, non-auxiliary, has trees → should survive.
        # Trees: one living (""), one dead ("2").
        trees1 = self._make_reference_trees(["", "2"])

        self.stand1 = Mock(
            spec=ForestStand,
            reference_trees=trees1,
            is_forest_land=Mock(return_value=True),
            is_auxiliary=Mock(return_value=False),
            has_trees=Mock(return_value=True),
            has_strata=Mock(return_value=False),
        )

        # stand2: non-forest land auxiliary with no trees → should be filtered out.
        empty_trees = ReferenceTrees()

        self.stand2 = Mock(
            spec=ForestStand,
            reference_trees=empty_trees,
            is_forest_land=Mock(return_value=False),
            is_auxiliary=Mock(return_value=True),
            has_trees=Mock(return_value=False),
            has_strata=Mock(return_value=False),
        )

        self.stands = StandList([self.stand1, self.stand2])

    def test_prepare_rst_output(self):
        result = prepare_rst_output(self.stands)

        # Only the forest-land, non-auxiliary stand should remain
        self.assertEqual(len(result), 1)

        # Only living trees should remain in the SoA container
        self.assertEqual(result[0].reference_trees.size, 1)

        # (optional) verify that the remaining tree_number indexing is 1..N
        self.assertListEqual(result[0].reference_trees.tree_number.tolist(), [1])

    def test_classify_values_to_valid_format(self):
        # Dummy data
        stand = ForestStand(
            geo_location=(6654200, 102598, 0.0, "EPSG:3067"),
            area_weight=100.0,
            auxiliary_stand=True)
        stand.reference_trees_pre_vec.append(
            ReferenceTree(species=TreeSpecies.SPRUCE, stand=stand))
        stand.tree_strata_pre_vec.append(
            TreeStratum(species=TreeSpecies.PINE, stand=stand))

        # fixture
        operation_params = {'format': 'rst'}

        # test
        result = classify_values_to([stand], **operation_params)
        self.assertEqual(len(result), 1)

    def test_classify_values_to_invalid_format(self):
        operation_params = {'format': 'invalid_format'}
        with self.assertRaises(ConfigurationException):
            classify_values_to(self.stands, **operation_params)
