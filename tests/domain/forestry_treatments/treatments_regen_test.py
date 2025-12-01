import unittest

from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.forestry_treatments.regeneration import regeneration

class RegenerationTest(unittest.TestCase):
    def make_stand(self, identifier="stand-A", year=2025):
        st = ForestStand()
        st.identifier = identifier
        st.year = year
        return st

    def test_artificial_regeneration_adds_trees_and_sets_year(self):
        stand = self.make_stand()
        params = {
            "origin": 2,
            "method": 2,
            "species": 1,
            "stems_per_ha": 1500.0,
            "height": 0.7,
            "biological_age": 3.0,
            "type": "artificial",
            "ntrees": 5,
        }
        start_size = stand.reference_trees.size
        updated, cdata = regeneration(stand, **params)

        self.assertEqual([], cdata)
        self.assertEqual(stand.year, updated.artificial_regeneration_year)

        self.assertEqual(start_size + params["ntrees"], updated.reference_trees.size)

        per_tree = params["stems_per_ha"] / params["ntrees"]
        new_idx = slice(-params["ntrees"], None)
        self.assertTrue(all(s == per_tree for s in updated.reference_trees.stems_per_ha[new_idx]))
        self.assertTrue(all(int(s) == params["species"] for s in updated.reference_trees.species[new_idx]))
        self.assertTrue(all(int(o) == params["origin"] for o in updated.reference_trees.origin[new_idx]))
        self.assertTrue(all(h == params["height"] for h in updated.reference_trees.height[new_idx]))
        self.assertTrue(all(a == params["biological_age"] for a in updated.reference_trees.biological_age[new_idx]))

    def test_natural_regeneration_does_not_set_artificial_year(self):
        stand = self.make_stand()
        params = {
            "origin": 1,
            "species": 2,
            "stems_per_ha": 900.0,
            "height": 0.6,
            "biological_age": 2.0,
            "type": "natural",
            "ntrees": 3,
        }
        start_size = stand.reference_trees.size
        updated, cdata = regeneration(stand, **params)

        self.assertEqual([], cdata)
        self.assertIsNone(updated.artificial_regeneration_year)
        self.assertEqual(start_size + params["ntrees"], updated.reference_trees.size)

        # verify only the 3 newly created rows
        new_idx = slice(-params["ntrees"], None)
        self.assertTrue(all(int(s) == params["species"] for s in updated.reference_trees.species[new_idx]))
        self.assertTrue(all(int(o) == params["origin"] for o in updated.reference_trees.origin[new_idx]))
        self.assertTrue(all(h == params["height"] for h in updated.reference_trees.height[new_idx]))
        self.assertTrue(all(a == params["biological_age"] for a in updated.reference_trees.biological_age[new_idx]))

    def test_identifiers_continue_after_existing_trees(self):
        stand = self.make_stand()
        # First regeneration adds 2 trees -> identifiers ...-1-tree, ...-2-tree
        regeneration(stand,
                     origin=2, species=1, stems_per_ha=200.0,
                     height=0.5, biological_age=1.0, type="artificial", ntrees=2)
        # Second regeneration adds 3 more -> identifiers start at 3
        updated, _ = regeneration(stand,
                                  origin=2, species=1, stems_per_ha=300.0,
                                  height=0.6, biological_age=2.0, type="artificial", ntrees=3)
        ids = list(updated.reference_trees.identifier)
        self.assertIn(f"{stand.identifier}-3-tree", ids)
        self.assertIn(f"{stand.identifier}-4-tree", ids)
        self.assertIn(f"{stand.identifier}-5-tree", ids)

    def test_optional_parameters_are_propagated(self):
        stand = self.make_stand()
        start_size = stand.reference_trees.size
        updated, _ = regeneration(
            stand,
            origin=2,
            species=1,
            stems_per_ha=1000.0,
            height=0.8,
            biological_age=4.0,
            type="artificial",
            ntrees=2,
            breast_height_diameter=1.2,
            breast_height_age=1.0,
        )
        self.assertEqual(start_size + 2, updated.reference_trees.size)
        new_idx = slice(-2, None)
        self.assertTrue(all(v == 1.2 for v in updated.reference_trees.breast_height_diameter[new_idx]))
        self.assertTrue(all(v == 1.0 for v in updated.reference_trees.breast_height_age[new_idx]))

    def test_validation_errors(self):
        stand = self.make_stand()

        with self.assertRaises(MetsiException):
            regeneration(stand,
                         origin=2, species=1, stems_per_ha=1000.0,
                         height=0.0, biological_age=3.0, type="artificial", ntrees=5)

        with self.assertRaises(MetsiException):
            regeneration(stand,
                         origin=2, species=1, stems_per_ha=1000.0,
                         height=0.5, biological_age=3.0, type="invalid", ntrees=5)

        with self.assertRaises(MetsiException):
            regeneration(stand,
                         origin=2, species=1, stems_per_ha=1000.0,
                         height=0.5, biological_age=3.0, type="natural", ntrees=0)

        with self.assertRaises(MetsiException):
            regeneration(stand,
                         origin=2, species=1, stems_per_ha=0.0,
                         height=0.5, biological_age=3.0, type="natural", ntrees=5)

if __name__ == "__main__":
    unittest.main()
