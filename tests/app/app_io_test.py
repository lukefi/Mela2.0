import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from lukefi.metsi.app.utils import ConfigurationException
from lukefi.metsi.app.app_io import (
    MetsiConfiguration, RunMode,
    StateFormat,
    parse_cli_arguments, generate_application_configuration
)

from lukefi.metsi.app.file_io import delete_existing_export_files


class TestAppIO(unittest.TestCase):

    def test_sim_cli_arguments(self):
        args = ['input.dat', 'out', 'control.py']
        result = parse_cli_arguments(args)
        self.assertEqual(4, len(result.keys()))
        self.assertEqual('input.dat', result['input_path'])
        self.assertEqual('out', result['target_directory'])
        self.assertEqual('control.py', result['control_file'])
        self.assertFalse(result['delete'])

    def test_control_configurations(self):
        args = ['cli_input', 'cli_output', 'cli_control.py']
        cli_args = parse_cli_arguments(args)

        # y is CLI-only flag and must not go into app config
        cli_args.pop("delete", None)
        control_args = {
            'run_modes': ['preprocess',
                          'simulate'],
            'multiprocessing': True}
        app_args = {**cli_args, **control_args}
        result = generate_application_configuration(app_args)
        self.assertEqual(args[0], result.input_path)
        self.assertEqual(args[1], result.target_directory)
        self.assertEqual(args[2], result.control_file)
        self.assertEqual(StateFormat.CSV, result.state_format)  # MetsiConfiguration default
        self.assertEqual([RunMode.PREPROCESS, RunMode.SIMULATE], result.run_modes)

    def test_control_config_with_invalid_values(self):
        control_args = {'asd123': 123}
        self.assertRaises(ConfigurationException, generate_application_configuration, control_args)


class TestRunModes(unittest.TestCase):
    def test_valid_typical_run_modes(self):
        modes = ['preprocess', 'export_prepro', 'simulate', 'postprocess', 'export']
        result = MetsiConfiguration._validate_and_sort_run_modes(modes)
        self.assertEqual(result, [RunMode.PREPROCESS, RunMode.EXPORT_PREPRO,
                         RunMode.SIMULATE, RunMode.POSTPROCESS, RunMode.EXPORT])

    def test_valid_run_modes_sorted(self):
        modes = ['simulate', 'PREPROCESS', 'export']
        result = MetsiConfiguration._validate_and_sort_run_modes(modes)
        self.assertEqual(result, [RunMode.PREPROCESS, RunMode.SIMULATE, RunMode.EXPORT])

    def test_valid_run_modes_with_dependencies(self):
        modes = ['simulate', 'postprocess', 'export']
        result = MetsiConfiguration._validate_and_sort_run_modes(modes)
        self.assertEqual(result, [RunMode.SIMULATE, RunMode.POSTPROCESS, RunMode.EXPORT])

    def test_invalid_run_mode(self):
        modes = ['simulate', 'invalid_mode']
        self.assertRaises(ConfigurationException,
                          MetsiConfiguration._validate_and_sort_run_modes,
                          modes)

    def test_export_prepro_without_preprocess(self):
        modes = ['export_prepro']
        self.assertRaises(ConfigurationException,
                          MetsiConfiguration._validate_and_sort_run_modes,
                          modes)

    def test_export_without_simulate_or_postprocess(self):
        modes = ['export']
        self.assertRaises(ConfigurationException,
                          MetsiConfiguration._validate_and_sort_run_modes,
                          modes)


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
