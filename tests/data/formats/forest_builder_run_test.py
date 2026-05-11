import unittest
from pathlib import Path
from lukefi.metsi.app.metsi_enum import StrataOrigin
from lukefi.metsi.data.formats.forest_centre.forest_centre_builder import GeoPackageBuilder, XMLBuilder
from lukefi.metsi.data.formats.nfi.vmi10_builder import VMI10Builder
from lukefi.metsi.data.formats.nfi.vmi11_builder import VMI11Builder
from lukefi.metsi.data.formats.nfi.vmi12_builder import VMI12Builder
from lukefi.metsi.data.formats.nfi.vmi13_builder import VMI13Builder
from lukefi.metsi.data.formats.nfi.vmi9_builder import VMI9Builder


def vmi_file_reader(file: Path) -> list[str]:
    with open(file, 'r', encoding='utf-8') as input_file:
        return input_file.readlines()


def xml_file_reader(file: Path) -> str:
    with open(file, 'r', encoding='utf-8') as input_file:
        return input_file.read()


class TestForestBuilderRun(unittest.TestCase):

    def test_run_smk_forest_builder_build(self):
        assertion = ('SMK_source.xml', 2)
        reference_file = Path('tests', 'data', 'resources', assertion[0])
        list_of_stands = XMLBuilder(
            builder_flags={"strata_origin": StrataOrigin.INVENTORY, "measured_trees": False},
            declared_conversions={},
            data=xml_file_reader(reference_file)).build()
        result = len(list_of_stands)
        self.assertEqual(result, assertion[1])

    def test_run_vmi12_forest_builder_build(self):
        assertion = ('VMI12_source_mini.dat', 4)
        reference_file = Path('tests', 'data', 'resources', assertion[0])
        list_of_stands = VMI12Builder(
            builder_flags={"measured_trees": False, "strata": True},
            declared_conversions={},
            data_rows=vmi_file_reader(reference_file)).build()
        result = len(list_of_stands)
        self.assertEqual(result, assertion[1])

    def test_run_vmi13_forest_builder_build(self):
        assertion = ('VMI13_source_mini.dat', 4)
        reference_file = Path('tests', 'data', 'resources', assertion[0])
        list_of_stands = VMI13Builder(
            builder_flags={"measured_trees": False, "strata": True},
            declared_conversions={},
            data_rows=vmi_file_reader(reference_file)).build()
        result = len(list_of_stands)
        self.assertEqual(result, assertion[1])

    def test_run_vmi11_forest_builder_build(self):
        assertion = ('VMI11_mini.dat', 3)
        reference_file = Path('tests', 'data', 'resources', assertion[0])
        list_of_stands = VMI11Builder(
            builder_flags={"measured_trees": False, "strata": True},
            declared_conversions={},
            data_rows=vmi_file_reader(reference_file)).build()
        result = len(list_of_stands)
        self.assertEqual(result, assertion[1])

    def test_run_vmi10_forest_builder_build(self):
        assertion = ('VMI10_mini.dat', 3)
        reference_file = Path('tests', 'data', 'resources', assertion[0])
        list_of_stands = VMI10Builder(
            builder_flags={"measured_trees": False, "strata": True},
            declared_conversions={},
            data_rows=vmi_file_reader(reference_file)).build()
        result = len(list_of_stands)
        self.assertEqual(result, assertion[1])

    def test_run_vmi9_forest_builder_build(self):
        assertion = ('VMI9_mini.dat', 3)
        reference_file = Path('tests', 'data', 'resources', assertion[0])
        list_of_stands = VMI9Builder(
            builder_flags={"measured_trees": False, "strata": True},
            declared_conversions={},
            data_rows=vmi_file_reader(reference_file)).build()
        result = len(list_of_stands)
        self.assertEqual(result, assertion[1])

    def test_run_smk_geopackage_builder_build(self):
        assertion = (('SMK_source.gpkg', 'geopackage'), 9)
        reference_file = Path('tests', 'data', 'resources', assertion[0][0])
        list_of_stands = GeoPackageBuilder(
            builder_flags={"strata_origin": StrataOrigin.INVENTORY},
            declared_conversions={},
            db_path=reference_file).build()
        result = len(list_of_stands)
        self.assertEqual(result, assertion[1])
