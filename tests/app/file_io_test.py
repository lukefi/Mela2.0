import unittest
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

import numpy as np
from lukefi.metsi.app import file_io
from lukefi.metsi.data.enums.internal import (
    DrainageCategory, LandUseCategory, OwnerCategory, SiteType,
    SoilPeatlandCategory, TreeSpecies)
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.app.app_types import ExportableContainer
from lukefi.metsi.app.app_io import MetsiConfiguration


@dataclass
class Test:
    a: int

    def __eq__(self, other):
        return self.a == other.a


class TestFileReading(unittest.TestCase):

    def test_determine_file_path(self):
        assertions = [
            (('testdir', 'preprocessing_result.rst'), (Path('testdir', 'preprocessing_result.rst'))),
            (('testdir', 'preprocessing_result.csv'), (Path('testdir', 'preprocessing_result.csv')))
        ]
        for a in assertions:
            result = file_io.determine_file_path(*a[0])
            self.assertEqual(a[1], result)

    def test_csv(self):
        stand = ForestStand(
            identifier="123-234",
            geo_location=(600000.0, 300000.0, 30.0, "EPSG:3067"),
        )

        # Add a couple of trees and one stratum to verify roundtrip correctness.
        stand.reference_trees.create([
            {
                "identifier": "123-234-1",
                "tree_number": 1,
                "species": int(TreeSpecies.PINE),
                "stems_per_ha": 200.0,
                "breast_height_diameter": 30.0,
                "height": 20.0,
                "sapling": False,
            },
            {
                "identifier": "123-234-2",
                "tree_number": 2,
                "species": int(TreeSpecies.SPRUCE),
                "stems_per_ha": 150.0,
                "breast_height_diameter": 25.0,
                "height": 17.0,
                "sapling": False,
            },
        ])
        stand.tree_strata.create([
            {
                "identifier": "123-234-S1",
                "species": int(TreeSpecies.PINE),
                "stems_per_ha": 500.0,
                "mean_diameter": 8.0,
                "mean_height": 6.0,
                "sapling_stratum": False,
            }
        ])

        data = [stand]
        ec = ExportableContainer(export_objects=data, additional_vars=None)

        file_io.prepare_target_directory("outdir")
        file_io.csv_writer(Path("outdir", "output.csv"), ec)
        result = file_io.csv_content_to_stands(file_io.csv_file_reader(Path("outdir/output.csv")))
        self.assertEqual(len(result), 1)
        result_stand = result[0]

        # Basic stand fields
        self.assertEqual(result_stand.identifier, stand.identifier)
        self.assertEqual(result_stand.geo_location, stand.geo_location)

        # ReferenceTrees (SoA)
        self.assertEqual(result_stand.reference_trees.size, stand.reference_trees.size)
        self.assertTrue(np.array_equal(result_stand.reference_trees.identifier, stand.reference_trees.identifier))
        self.assertTrue(np.array_equal(result_stand.reference_trees.tree_number, stand.reference_trees.tree_number))
        self.assertTrue(np.array_equal(result_stand.reference_trees.species, stand.reference_trees.species))
        self.assertTrue(np.allclose(result_stand.reference_trees.stems_per_ha, stand.reference_trees.stems_per_ha))
        self.assertTrue(
            np.allclose(
                result_stand.reference_trees.breast_height_diameter,
                stand.reference_trees.breast_height_diameter))
        self.assertTrue(np.allclose(result_stand.reference_trees.height, stand.reference_trees.height))
        self.assertTrue(np.array_equal(result_stand.reference_trees.sapling, stand.reference_trees.sapling))

        # TreeStrata (SoA)
        self.assertEqual(result_stand.tree_strata.size, stand.tree_strata.size)
        self.assertTrue(np.array_equal(result_stand.tree_strata.identifier, stand.tree_strata.identifier))
        self.assertTrue(np.array_equal(result_stand.tree_strata.species, stand.tree_strata.species))
        self.assertTrue(np.allclose(result_stand.tree_strata.stems_per_ha, stand.tree_strata.stems_per_ha))
        self.assertTrue(np.allclose(result_stand.tree_strata.mean_diameter, stand.tree_strata.mean_diameter))
        self.assertTrue(np.allclose(result_stand.tree_strata.mean_height, stand.tree_strata.mean_height))
        shutil.rmtree('outdir')

    def test_rst(self):
        data = [
            ForestStand(
                identifier="123-234",
                geo_location=(600000.0, 300000.0, 30.0, "EPSG:3067"),
                land_use_category=LandUseCategory.FOREST,
                owner_category=OwnerCategory.PRIVATE,
                soil_peatland_category=SoilPeatlandCategory.MINERAL_SOIL,
                site_type_category=SiteType.RICH_SITE,
                drainage_category=DrainageCategory.DITCHED_MINERAL_SOIL,
                fra_category="1",
                auxiliary_stand=False,
                time=2025,
            )
        ]

        data[0].reference_trees.create([
            {
                "identifier": "123-234-1",
                "species": int(TreeSpecies.PINE),
                "stems_per_ha": 200.0,
                "sapling": False,
            }
        ])
        ec = ExportableContainer(export_objects=data, additional_vars=None)

        file_io.prepare_target_directory("outdir")
        target = Path("outdir", "output.rst")
        file_io.rst_writer(target, ec)

        # There is no rst input so check sanity just by file existence and non-emptiness
        exists = os.path.exists(target)
        size = os.path.getsize(target)
        self.assertTrue(exists)
        self.assertTrue(size > 0)
        shutil.rmtree('outdir')

    def test_read_stands_from_csv_file(self):
        config = MetsiConfiguration(
            input_path="tests/resources/file_io_test/forest_centre.csv",
            state_format="csv",
        )
        stands_from_csv = file_io.read_stands_from_file(config, {})
        self.assertEqual(len(stands_from_csv), 2)
        self.assertEqual(type(stands_from_csv[0]), ForestStand)
        self.assertIsInstance(stands_from_csv[0].reference_trees, ReferenceTrees)
        self.assertIsInstance(stands_from_csv[0].tree_strata, TreeStrata)

    def test_read_stands_from_vmi12_file(self):
        config = MetsiConfiguration(
            input_path="tests/resources/file_io_test/vmi12.dat",
            state_format="vmi12",
        )
        stands = file_io.read_stands_from_file(config, {})
        self.assertEqual(len(stands), 4)

    def test_read_stands_from_vmi13_file(self):
        config = MetsiConfiguration(
            input_path=Path("tests", "data", "resources", "VMI13_source_mini.dat"),
            state_format="vmi13",
        )
        stands = file_io.read_stands_from_file(config, {})
        self.assertEqual(len(stands), 4)

    def test_read_stands_from_xml_file(self):
        config = MetsiConfiguration(
            input_path="tests/resources/file_io_test/forest_centre.xml",
            state_format="xml",
        )
        stands = file_io.read_stands_from_file(config, {})
        self.assertEqual(len(stands), 2)

    def test_read_stands_from_gpkg_file(self):
        config = MetsiConfiguration(
            input_path="tests/data/resources/SMK_source.gpkg",
            state_format="gpkg",
        )
        stands = file_io.read_stands_from_file(config, {})
        self.assertEqual(len(stands), 9)

    def test_read_stands_from_nonexisting_file(self):
        config = MetsiConfiguration(
            input_path="nonexisting_file.pickle",
            state_format="csv",
        )
        self.assertRaises(Exception, file_io.read_stands_from_file, config)


class TestReadControlModule(unittest.TestCase):
    @patch("importlib.util.spec_from_file_location")
    @patch("importlib.util.module_from_spec")
    def test_read_control_module_success(self, mock_module_from_spec, mock_spec_from_file_location):
        # Mock the control module
        mock_spec = MagicMock()
        mock_loader = MagicMock()
        mock_spec.loader = mock_loader
        mock_spec_from_file_location.return_value = mock_spec

        mock_module = MagicMock()
        mock_module.control_structure = {"key": "value"}
        mock_module_from_spec.return_value = mock_module

        # Call the function
        control_path = str(Path("test_control.py").resolve())  # Resolve to absolute path
        result = file_io.read_control_module(control_path, "control_structure")

        # Assertions
        mock_spec_from_file_location.assert_called_once_with("test_control", control_path)
        mock_loader.exec_module.assert_called_once_with(mock_module)
        self.assertEqual(result, {"key": "value"})

    def test_read_control_module_attribute_error(self):
        control_path = os.path.join(os.getcwd(), "tests", "resources", "file_io_test", "dummy_control.py")
        with open(control_path, "w", encoding="utf-8") as f:
            f.write("some_variable = {'key': 'value'}")  # Create a dummy control file without the expected attribute

        with self.assertRaises(AttributeError) as context:
            file_io.read_control_module(control_path, control="nonexistent_control")

        self.assertIn("Variable 'nonexistent_control' not found", str(context.exception))
        os.remove(control_path)

    @patch("importlib.util.spec_from_file_location")
    def test_read_control_module_import_error(self, mock_spec_from_file_location):
        # Simulate an import error
        mock_spec_from_file_location.return_value = None

        # Call the function and expect an ImportError
        control_path = str(Path("test_control.py").resolve())  # Resolve to absolute path
        with self.assertRaises(ImportError):
            file_io.read_control_module(control_path, "control_structure")
