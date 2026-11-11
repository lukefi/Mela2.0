from typing import cast

import numpy as np
import unittest

from lukefi.metsi.core.utils.select_units import Mode, ProfileXMode, SelectionSet, SelectionTarget, TargetType
from lukefi.metsi.data.enums.internal import OwnerCategory, Storey
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.domain.collected_data import RemovedTrees
from lukefi.metsi.domain.forestry_treatments.remove_trees import remove_trees_fn


class RemoveTreesTest(unittest.TestCase):

    def test_remove_trees(self):
        stand = ForestStand()
        stand.identifier = "test"
        stand.year = 2030

        rt = ReferenceTrees()
        rt.vectorize({
            "identifier": [f"test-{i}-tree" for i in range(1, 4)],
            "stems_per_ha": [100.0, 200.0, 300.0],
            "height": [10.0, 20.0, 30.0],
            "species": [1, 1, 2],
            "tree_number": [1, 2, 3],
            "storey": [Storey.DOMINANT, Storey.DOMINANT, Storey.DOMINANT]
        })
        stand.reference_trees = rt

        total = np.sum(stand.reference_trees.stems_per_ha)

        before = stand.reference_trees.stems_per_ha.copy()
        after, cdata = remove_trees_fn(
            stand,
            tree_selection={
                "target": SelectionTarget(TargetType.ABSOLUTE, "stems_per_ha", 0.5 * total),
                "sets": [
                    SelectionSet[ForestStand, ReferenceTrees](
                        lambda _, trees: np.repeat(True, trees.size),
                        "height",
                        "stems_per_ha",
                        TargetType.ABSOLUTE,
                        0.5 * total,
                        [0.0, 1.0],
                        [0.5, 0.5],
                        ProfileXMode.RELATIVE
                    )
                ]
            },
            mode=Mode.ODDS_UNITS,
            select_from_all=True,
            updated_attributes={
                "owner_category": OwnerCategory.CONGREGATION
            })

        self.assertEqual(1, len(cdata))
        self.assertIsInstance(cdata[0], RemovedTrees)
        removed_trees = cast(RemovedTrees, cdata[0]).removed_trees

        np.testing.assert_allclose(removed_trees.stems_per_ha, before * 0.5, rtol=0, atol=1e-9)
        self.assertEqual(OwnerCategory.CONGREGATION, after.owner_category)
        np.testing.assert_allclose(before * 0.5, after.reference_trees.stems_per_ha, rtol=0, atol=1e-9)
