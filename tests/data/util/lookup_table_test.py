import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from lukefi.metsi.data.util.lookup_table import LookupTable
from lukefi.metsi.domain.domain_tables import min_stems_table
from lukefi.metsi.data.enums.internal import SiteType
from lukefi.metsi.data.model import ForestStand


@dataclass
class DummyStand:
    # Must match LookupTable key_columns by attribute name
    site_type_category: int
    development_class: int
    degree_days: int


class TestLookupTableMinStems(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Same default path your factory uses
        cls.csv_path = Path("data") / "parameter_files" / "min_stems.csv"
        if not cls.csv_path.exists():
            raise FileNotFoundError(
                f"min_stems.csv not found at {cls.csv_path}. "
                "Make sure it is included in the repo and tests run from repo root."
            )

        cls.table = min_stems_table(cls.csv_path)

    def test_lookup_returns_expected_value_for_known_row(self):
        # Your CSV row is: 2,1,2,930
        # BUT transforms apply:
        # - site_type_category -> site_group_for(v) returns 2 only when v == DAMP_SITE.value
        # - degree_days -> dd_group_for(degree_days) returns 2 when 1200 <= degree_days < 1400
        stand = DummyStand(
            site_type_category=SiteType.DAMP_SITE.value,  # maps to key "2"
            development_class=999,                        # ignored by species_group_for anyway
            degree_days=1300,                             # maps to key "2"
        )

        value = self.table(cast(ForestStand, stand))
        self.assertEqual(value, 930)

    def test_lookup_raises_valueerror_for_missing_key(self):
        # combination not in csv.
        stand = DummyStand(
            site_type_category=SiteType.OPEN_MOUNTAINS.value,  # key "4"
            development_class=999,                        # key "1" via species_group_for
            degree_days=1300,                             # key "1" via dd_group_for
        )

        with self.assertRaises(ValueError) as cm:
            _ = self.table(cast(ForestStand, stand))

        msg = str(cm.exception)
        self.assertIn("No matching row in CSV", msg)
        self.assertIn("site_type_category", msg)
        self.assertIn("development_class", msg)
        self.assertIn("degree_days", msg)

    def test_site_group_for_rejects_unsupported_values(self):
        # site_group_for raises if outside expected 1..8
        stand = DummyStand(
            site_type_category=999,   # should raise
            development_class=1,
            degree_days=1300,
        )
        with self.assertRaises(ValueError):
            _ = self.table(cast(ForestStand, stand))


class LookupTableTest(unittest.TestCase):

    def test_external_keys(self):
        lut = LookupTable[ForestStand, int](
            "tests/data/resources/lut_test.csv",
            key_columns=["site_type_category", "species_to_be_planted"],
            value_column="min_seedlings",
            value_cast=int,
            key_transforms={"site_type_category": int}
        )

        stand = ForestStand(site_type_category=SiteType.VERY_RICH_SITE)

        self.assertEqual(lut(stand, species_to_be_planted="pine"), 2400)
        self.assertEqual(lut(stand, species_to_be_planted="spruce"), 2000)
        self.assertEqual(lut(stand, species_to_be_planted="silver_birch"), 1600)
        self.assertEqual(lut(stand, species_to_be_planted="downy_birch"), 1600)
        self.assertEqual(lut(stand, species_to_be_planted="other_deciduous"), 1600)
