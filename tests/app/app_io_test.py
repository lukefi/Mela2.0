import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch


from lukefi.metsi.app.app_io import parse_cli_arguments
from lukefi.metsi.app.file_io import delete_existing_export_files
from lukefi.metsi.app.metsi_control import AppConfiguration
from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.sim.exceptions import ConfigurationException


class TestAppIO(unittest.TestCase):

    def test_sim_cli_arguments(self):
        args = ['input.dat', 'out', 'control.py']
        result = parse_cli_arguments(args)
        self.assertEqual(4, len(result.keys()))
        self.assertEqual('input.dat', result['input_path'])
        self.assertEqual('out', result['target_directory'])
        self.assertEqual('control.py', result['control_file'])
        self.assertFalse(result['delete'])


class TestRunModes(unittest.TestCase):
    def test_valid_typical_run_modes(self):
        _ = AppConfiguration(
            state_format=StateFormat.VMI13,
            run_modes=[
                RunMode.PREPROCESS,
                RunMode.EXPORT_PREPRO,
                RunMode.SIMULATE,
            ])

    def test_export_prepro_without_preprocess(self):
        self.assertRaises(ConfigurationException, AppConfiguration, state_format=StateFormat.VMI13, run_modes=[
            RunMode.EXPORT_PREPRO,
        ])


class TestDeleteExistingExportFiles(unittest.TestCase):

    def test_no_existing_files_returns_true_and_no_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            with patch("builtins.input") as mock_input:
                ok = delete_existing_export_files(
                    target_directory=td,
                    export_prepro={"csv": {}, "rst": {}},
                    preprocessing_base_name="pre",
                    simulation_base_name="sim",
                    force_delete=False,
                )
                self.assertTrue(ok)
                mock_input.assert_not_called()

    def test_existing_files_user_says_no_returns_false_and_does_not_delete(self):
        with tempfile.TemporaryDirectory() as td:
            # Create files that should be detected as existing
            sim_db = Path(td) / "sim.db"
            pre_csv = Path(td) / "pre.csv"
            pre_rst = Path(td) / "pre.rst"
            par = Path(td) / "c-variables.par"
            for p in (sim_db, pre_csv, pre_rst, par):
                p.write_text("dummy", encoding="utf-8")

            with patch("builtins.input", return_value="n") as mock_input:
                ok = delete_existing_export_files(
                    target_directory=td,
                    export_prepro={"csv": {}, "rst": {}},
                    preprocessing_base_name="pre",
                    simulation_base_name="sim",
                    force_delete=False,
                )
                self.assertFalse(ok)
                mock_input.assert_called_once()

            # Ensure files still exist
            self.assertTrue(sim_db.exists())
            self.assertTrue(pre_csv.exists())
            self.assertTrue(pre_rst.exists())
            self.assertTrue(par.exists())

    def test_existing_files_user_says_yes_deletes_and_returns_true(self):
        with tempfile.TemporaryDirectory() as td:
            sim_db = Path(td) / "sim.db"
            pre_csv = Path(td) / "pre.csv"
            pre_rst = Path(td) / "pre.rst"
            par = Path(td) / "c-variables.par"
            for p in (sim_db, pre_csv, pre_rst, par):
                p.write_text("dummy", encoding="utf-8")

            with patch("builtins.input", return_value="y") as mock_input:
                ok = delete_existing_export_files(
                    target_directory=td,
                    export_prepro={"csv": {}, "rst": {}},
                    preprocessing_base_name="pre",
                    simulation_base_name="sim",
                    force_delete=False,
                )
                self.assertTrue(ok)
                mock_input.assert_called_once()

            # Ensure files were deleted
            self.assertFalse(sim_db.exists())
            self.assertFalse(pre_csv.exists())
            self.assertFalse(pre_rst.exists())
            self.assertFalse(par.exists())

    def test_force_delete_true_skips_prompt_and_deletes(self):
        with tempfile.TemporaryDirectory() as td:
            sim_db = Path(td) / "sim.db"
            pre_csv = Path(td) / "pre.csv"
            sim_db.write_text("dummy", encoding="utf-8")
            pre_csv.write_text("dummy", encoding="utf-8")

            with patch("builtins.input") as mock_input:
                ok = delete_existing_export_files(
                    target_directory=td,
                    export_prepro={"csv": {}},
                    preprocessing_base_name="pre",
                    simulation_base_name="sim",
                    force_delete=True,
                )
                self.assertTrue(ok)
                mock_input.assert_not_called()

            self.assertFalse(sim_db.exists())
            self.assertFalse(pre_csv.exists())

    def test_respects_custom_base_names_from_control_file(self):
        # This matches your control_structure override idea:
        # preprocessing_output_file="pre", simulation_output_file="sim"
        with tempfile.TemporaryDirectory() as td:
            # Use NON-default names to prove the function uses the parameters
            sim_db = Path(td) / "customsim.db"
            pre_csv = Path(td) / "custompre.csv"
            sim_db.write_text("dummy", encoding="utf-8")
            pre_csv.write_text("dummy", encoding="utf-8")

            with patch("builtins.input", return_value="yes"):
                ok = delete_existing_export_files(
                    target_directory=td,
                    export_prepro={"csv": {}},
                    preprocessing_base_name="custompre",
                    simulation_base_name="customsim",
                    force_delete=False,
                )
                self.assertTrue(ok)

            self.assertFalse(sim_db.exists())
            self.assertFalse(pre_csv.exists())
