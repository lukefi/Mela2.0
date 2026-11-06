import unittest

from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.enums.internal import Origin, TreeSpecies
from lukefi.metsi.domain.forestry_treatments.regeneration import RegenerationDecl, regeneration_fn
from lukefi.metsi.core.exceptions import MetsiException


class RegenerationTest(unittest.TestCase):
    def make_stand(self, identifier="stand-A", year=2025):
        st = ForestStand()
        st.identifier = identifier
        st.year = year
        return st

    def test_artificial_regeneration_adds_trees_and_sets_year(self):
        stand = self.make_stand()
        params = {
            "regeneration_decls": [
                RegenerationDecl(
                    method=Origin.PLANTED,
                    species=TreeSpecies.PINE,
                    stems_per_ha=1500.0,
                    height=0.7,
                    biological_age=3.0,
                    ntrees=5,
                )
            ]
        }
        start_size = stand.reference_trees.size
        updated, cdata = regeneration_fn(stand, **params)

        self.assertEqual([], cdata)
        self.assertEqual(stand.year, updated.artificial_regeneration_year)

        self.assertEqual(start_size + 5, updated.reference_trees.size)

        per_tree = 1500 / 5
        new_idx = slice(-5, None)
        self.assertTrue(all(s == per_tree for s in updated.reference_trees.stems_per_ha[new_idx]))
        self.assertTrue(all(int(s) == TreeSpecies.PINE for s in updated.reference_trees.species[new_idx]))
        self.assertTrue(all(int(o) == Origin.PLANTED for o in updated.reference_trees.origin[new_idx]))
        self.assertTrue(all(h == 0.7 for h in updated.reference_trees.height[new_idx]))
        self.assertTrue(all(a == 3.0 for a in updated.reference_trees.biological_age[new_idx]))

    def test_natural_regeneration_does_not_set_artificial_year(self):
        stand = self.make_stand()
        params = {
            "regeneration_decls": [RegenerationDecl(
                method=Origin.NATURAL,
                species=TreeSpecies.SPRUCE,
                stems_per_ha=900.0,
                height=0.6,
                biological_age=2.0,
                ntrees=3,)]
        }
        start_size = stand.reference_trees.size
        updated, cdata = regeneration_fn(stand, **params)

        self.assertEqual([], cdata)
        self.assertIsNone(updated.artificial_regeneration_year)
        self.assertEqual(start_size + 3, updated.reference_trees.size)

        # verify only the 3 newly created rows
        new_idx = slice(-3, None)
        self.assertTrue(all(int(s) == TreeSpecies.SPRUCE for s in updated.reference_trees.species[new_idx]))
        self.assertTrue(all(int(o) == Origin.NATURAL for o in updated.reference_trees.origin[new_idx]))
        self.assertTrue(all(h == 0.6 for h in updated.reference_trees.height[new_idx]))
        self.assertTrue(all(a == 2.0 for a in updated.reference_trees.biological_age[new_idx]))

    def test_identifiers_continue_after_existing_trees(self):
        stand = self.make_stand()
        # First regeneration adds 2 trees -> identifiers ...-1-tree, ...-2-tree
        regeneration_fn(stand,
                        regeneration_decls=[RegenerationDecl(method=Origin.PLANTED,
                                                             species=TreeSpecies.PINE,
                                                             stems_per_ha=200.0,
                                                             height=0.5,
                                                             biological_age=1.0,
                                                             ntrees=2)])
        # Second regeneration adds 3 more -> identifiers start at 3
        updated, _ = regeneration_fn(stand,
                                     regeneration_decls=[RegenerationDecl(method=Origin.PLANTED,
                                                                          species=TreeSpecies.PINE,
                                                                          stems_per_ha=300.0,
                                                                          height=0.6,
                                                                          biological_age=2.0,
                                                                          ntrees=3)])
        ids = list(updated.reference_trees.identifier)
        self.assertIn(f"{stand.identifier}-3-tree", ids)
        self.assertIn(f"{stand.identifier}-4-tree", ids)
        self.assertIn(f"{stand.identifier}-5-tree", ids)

    def test_optional_parameters_are_propagated(self):
        stand = self.make_stand()
        start_size = stand.reference_trees.size
        updated, _ = regeneration_fn(stand,
                                     regeneration_decls=[RegenerationDecl(method=Origin.PLANTED,
                                                                          species=TreeSpecies.PINE,
                                                                          stems_per_ha=1000.0,
                                                                          height=0.8,
                                                                          biological_age=4.0,
                                                                          ntrees=2,
                                                                          breast_height_diameter=1.2,
                                                                          breast_height_age=1.0)],)
        self.assertEqual(start_size + 2, updated.reference_trees.size)
        new_idx = slice(-2, None)
        self.assertTrue(all(v == 1.2 for v in updated.reference_trees.breast_height_diameter[new_idx]))
        self.assertTrue(all(v == 1.0 for v in updated.reference_trees.breast_height_age[new_idx]))


if __name__ == "__main__":
    unittest.main()
