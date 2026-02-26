import unittest
import numpy as np

from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.data.enums.internal import TreeSpecies
from lukefi.metsi.domain.utils.filter import filter_stands, filter_strata, filter_trees
from lukefi.metsi.domain.pre_ops import filter_stands as filter_stands_
from lukefi.metsi.domain.pre_ops import filter_trees as filter_trees_


class FilterTest(unittest.TestCase):

    def test_filter_stands(self):
        s900 = ForestStand(identifier="1", degree_days=900)
        s1000 = ForestStand(identifier="2", degree_days=1000)
        s1100 = ForestStand(identifier="3", degree_days=1100)

        self.assertEqual(
            filter_stands(
                [s900, s1000, s1100],
                "select",
                lambda stand: stand.degree_days is not None and stand.degree_days > 1050,
            ),
            [s1100],
        )
        self.assertEqual(
            filter_stands(
                [s900, s1000, s1100],
                "remove",
                lambda stand: stand.degree_days is not None and stand.degree_days > 1050,
            ),
            [s900, s1000],
        )

    def test_filter_trees_and_strata_soa(self):
        # Stand 1
        stand1 = ForestStand(identifier="S-1")
        stand1.reference_trees = ReferenceTrees().vectorize(
            {
                "identifier": ["t-1", "t-2", "t-3"],
                "species": [
                    int(TreeSpecies.PINE),
                    int(TreeSpecies.SPRUCE),
                    int(TreeSpecies.SILVER_BIRCH),
                ],
                "breast_height_diameter": [0.0, 0.0, 20.0],
                "height": [0.7, 0.6, 25.0],
            }
        )
        stand1.tree_strata = TreeStrata().vectorize(
            {
                "identifier": ["s-1", "s-2"],
                "species": [int(TreeSpecies.PINE), int(TreeSpecies.SPRUCE)],
            }
        )

        # Stand 2
        stand2 = ForestStand(identifier="S-2")
        stand2.reference_trees = ReferenceTrees().vectorize(
            {
                "identifier": ["t-4", "t-5"],
                "species": [int(TreeSpecies.GREY_ALDER), int(TreeSpecies.ASPEN)],
                "breast_height_diameter": [10.0, 15.0],
                "height": [15.0, 18.0],
            }
        )
        stand2.tree_strata = TreeStrata().vectorize(
            {
                "identifier": [],
                "species": [],
            }
        )

        # Remove small pine seedlings (<1.3m)
        filter_trees([stand1, stand2], lambda stand: ~((stand.reference_trees.height < 1.3)
                     & (stand.reference_trees.species == int(TreeSpecies.PINE))), )
        self.assertListEqual(stand1.reference_trees.identifier.tolist(), ["t-2", "t-3"])
        self.assertListEqual(stand2.reference_trees.identifier.tolist(), ["t-4", "t-5"])

        # Select very tall trees (>20m)
        filter_trees(
            [stand1, stand2],
            lambda stand: stand.reference_trees.height > 20.0,
        )
        self.assertListEqual(stand1.reference_trees.identifier.tolist(), ["t-3"])
        self.assertEqual(stand2.reference_trees.size, 0)

        # Select spruce strata
        filter_strata(
            [stand1, stand2],
            lambda stand: stand.tree_strata.species == int(TreeSpecies.SPRUCE),
        )
        self.assertListEqual(stand1.tree_strata.identifier.tolist(), ["s-2"])
        self.assertEqual(stand2.tree_strata.size, 0)

    def test_filter_named(self):
        s1 = ForestStand(identifier="1")
        s2 = ForestStand(identifier="2")
        s3 = ForestStand(identifier="3")

        self.assertEqual(
            filter_stands(
                [s1, s2, s3],
                "select",
                lambda stand: stand.identifier in ["1", "3"],
            ),
            [s1, s3],
        )

    def test_reject_invalid_command(self):
        with self.assertRaises(ValueError):
            filter_stands([], "? ? ?", lambda x: 1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            filter_stands([], "choose", lambda x: 1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            filter_stands([], "select something", lambda x: 1)  # type: ignore[arg-type]

    def test_preproc_filter(self):
        s1 = ForestStand(identifier="1")
        s1.reference_trees = ReferenceTrees().vectorize(
            {
                "identifier": ["1", "2", "3"],
                "species": [int(TreeSpecies.PINE), int(TreeSpecies.PINE), int(TreeSpecies.PINE)],
            }
        )

        s2 = ForestStand(identifier="2")
        s2.reference_trees = ReferenceTrees().vectorize(
            {
                "identifier": ["4"],
                "species": [int(TreeSpecies.SPRUCE)],
            }
        )

        stands = filter_trees_(
            [s1, s2],
            **{
                "predicate": lambda stand: ~np.isin(stand.reference_trees.identifier, ["3", "4"])
            }
        )
        stands = filter_stands_(
            stands,
            **{
                "select": lambda stand: stand.reference_trees.size > 0,
            }
        )

        self.assertEqual(stands, [s1])
        self.assertListEqual(s1.reference_trees.identifier.tolist(), ["1", "2"])
        self.assertEqual(s2.reference_trees.size, 0)
