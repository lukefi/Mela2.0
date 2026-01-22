import unittest
import numpy as np

from lukefi.metsi.domain.utils.filter import applyfilter
from lukefi.metsi.domain.pre_ops import preproc_filter

from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.data.enums.internal import TreeSpecies


class FilterTest(unittest.TestCase):

    def test_filter_stands(self):
        s900 = ForestStand(identifier="1", degree_days=900)
        s1000 = ForestStand(identifier="2", degree_days=1000)
        s1100 = ForestStand(identifier="3", degree_days=1100)

        self.assertEqual(
            applyfilter(
                [s900, s1000, s1100],
                "select stands",
                lambda stand: stand.degree_days > 1050,
            ),
            [s1100],
        )
        self.assertEqual(
            applyfilter(
                [s900, s1000, s1100],
                "remove stands",
                lambda stand: stand.degree_days > 1050,
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
        applyfilter(
            [stand1, stand2],
            "remove trees",
            lambda trees: (trees.height < 1.3) & (trees.species == int(TreeSpecies.PINE)),
        )
        self.assertListEqual(stand1.reference_trees.identifier.tolist(), ["t-2", "t-3"])
        self.assertListEqual(stand2.reference_trees.identifier.tolist(), ["t-4", "t-5"])

        # Select very tall trees (>20m)
        applyfilter(
            [stand1, stand2],
            "select trees",
            lambda trees: trees.height > 20.0,
        )
        self.assertListEqual(stand1.reference_trees.identifier.tolist(), ["t-3"])
        self.assertEqual(stand2.reference_trees.size, 0)

        # Select spruce strata
        applyfilter(
            [stand1, stand2],
            "select strata",
            lambda strata: strata.species == int(TreeSpecies.SPRUCE),
        )
        self.assertListEqual(stand1.tree_strata.identifier.tolist(), ["s-2"])
        self.assertEqual(stand2.tree_strata.size, 0)

    def test_filter_named(self):
        s1 = ForestStand(identifier="1")
        s2 = ForestStand(identifier="2")
        s3 = ForestStand(identifier="3")

        self.assertEqual(
            applyfilter(
                [s1, s2, s3],
                "select",
                lambda stand: stand.identifier in ["1", "3"],
            ),
            [s1, s3],
        )

    def test_reject_invalid_command(self):
        with self.assertRaisesRegex(ValueError, "filter syntax error"):
            applyfilter([], "? ? ?", lambda x: 1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "invalid filter verb"):
            applyfilter([], "choose", lambda x: 1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "invalid filter object"):
            applyfilter([], "select something", lambda x: 1)  # type: ignore[arg-type]

    def test_tree_predicate_must_return_correct_mask_shape(self):
        stand = ForestStand(identifier="S")
        stand.reference_trees = ReferenceTrees().vectorize(
            {
                "identifier": ["t1", "t2"],
                "species": [int(TreeSpecies.PINE), int(TreeSpecies.SPRUCE)],
            }
        )

        with self.assertRaisesRegex(ValueError, r"tree predicate must return mask of shape \(2,\)"):
            applyfilter([stand], "select trees", lambda trees: np.array([True]))  # type: ignore[arg-type]

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

        stands = preproc_filter(
            [s1, s2],
            **{
                "remove trees": lambda trees: np.isin(trees.identifier, ["3", "4"]),
                "select": lambda stand: stand.reference_trees.size > 0,
            },
        )

        self.assertEqual(stands, [s1])
        self.assertListEqual(s1.reference_trees.identifier.tolist(), ["1", "2"])
        self.assertEqual(s2.reference_trees.size, 0)
