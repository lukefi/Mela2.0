import unittest
from pathlib import Path
from lukefi.metsi.domain.domain_tables import min_stems_table


class MinStemsLookupTableTest(unittest.TestCase):
    def test_min_stems_lookup_uses_csv_and_transforms(self):
        """
        Uses the table in data\\parameter_files\\min_stems.csv.

        Expected CSV content:

            site_type_category  species_group  degree_days  min_stems
            1                   1              1            1250
            1                   2              1            1100
            1                   1              2            1200
            2                   1              1            1000
            2                   2              1            950

        With the current transforms in min_stems_table:

          - site_group_for(1) -> 1
          - dd_group_for(1000) -> 1
          - species_group_for(...) -> 1

        So a stand with (site_type_category=1, species=anything, degree_days=1000)
        should hit the first row and return min_stems=1250.
        """

        # Build the lookup table using the real CSV
        table = min_stems_table(Path("data") / "parameter_files" / "min_stems.csv")

        # Minimal fake stand object with the attributes LookupTable expects
        class FakeStand:
            def __init__(self, site_type_category, development_class, degree_days):
                self.site_type_category = site_type_category
                self.development_class = development_class
                self.degree_days = degree_days

        # Values chosen so that the transform functions map to (1, 1, 1)
        stand = FakeStand(
            site_type_category=1,  # site_group_for(1) -> 1
            development_class=1,             # species_group_for(...) -> 1 (ignores value)
            degree_days=1000,      # dd_group_for(1000) -> 1
        )

        result = table(stand)

        # From the first row of the example CSV: min_stems == 1250
        self.assertEqual(1250, result)
