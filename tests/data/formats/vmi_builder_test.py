import tempfile
import unittest
from pathlib import Path

from lukefi.metsi.app.app_types import ExportableContainer
from lukefi.metsi.app.file_io import csv_exp_writer
from tests.data.test_util import ForestBuilderTestBench
from tests.data.snapshot_util import assert_snapshot, assert_file_snapshot


class TestForestBuilderSnapshots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vmi9_stands = ForestBuilderTestBench.vmi9_built()
        cls.vmi10_stands = ForestBuilderTestBench.vmi10_built()
        cls.vmi11_stands = ForestBuilderTestBench.vmi11_built()
        cls.vmi12_stands = ForestBuilderTestBench.vmi12_built()
        cls.vmi13_stands = ForestBuilderTestBench.vmi13_built()

    def test_snapshot_vmi9(self):
        assert_snapshot(self, name="vmi9", stands=self.vmi9_stands)

    def test_snapshot_vmi10(self):
        assert_snapshot(self, name="vmi10", stands=self.vmi10_stands)

    def test_snapshot_vmi11(self):
        assert_snapshot(self, name="vmi11", stands=self.vmi11_stands)

    def test_snapshot_vmi12(self):
        assert_snapshot(self, name="vmi12", stands=self.vmi12_stands)

    def test_snapshot_vmi13(self):
        assert_snapshot(self, name="vmi13", stands=self.vmi13_stands)


class TestForestBuilderCsvExpSnapshots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vmi9_stands = ForestBuilderTestBench.vmi9_built()
        cls.vmi10_stands = ForestBuilderTestBench.vmi10_built()
        cls.vmi11_stands = ForestBuilderTestBench.vmi11_built()
        cls.vmi12_stands = ForestBuilderTestBench.vmi12_built()
        cls.vmi13_stands = ForestBuilderTestBench.vmi13_built()

    def _assert_csv_exp_snapshots(self, base_name: str, stands) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            # csv_exp_writer ignores the given filename itself and writes:
            #   stands.csv, trees.csv, strata.csv
            csv_exp_writer(
                out_dir / "preprocessing_result.csv_exp",
                ExportableContainer(export_objects=stands, additional_vars=None),
            )

            assert_file_snapshot(
                self,
                snapshot_name=f"{base_name}.stands.csv",
                actual_file=out_dir / "stands.csv",
            )
            assert_file_snapshot(
                self,
                snapshot_name=f"{base_name}.trees.csv",
                actual_file=out_dir / "trees.csv",
            )
            assert_file_snapshot(
                self,
                snapshot_name=f"{base_name}.strata.csv",
                actual_file=out_dir / "strata.csv",
            )

    def test_csv_exp_vmi9(self):
        self._assert_csv_exp_snapshots("vmi9", self.vmi9_stands)

    def test_csv_exp_vmi10(self):
        self._assert_csv_exp_snapshots("vmi10", self.vmi10_stands)

    def test_csv_exp_vmi11(self):
        self._assert_csv_exp_snapshots("vmi11", self.vmi11_stands)

    def test_csv_exp_vmi12(self):
        self._assert_csv_exp_snapshots("vmi12", self.vmi12_stands)

    def test_csv_exp_vmi13(self):
        self._assert_csv_exp_snapshots("vmi13", self.vmi13_stands)


class TestForestBuilder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vmi9_builder = ForestBuilderTestBench.vmi9_builder
        cls.vmi10_builder = ForestBuilderTestBench.vmi10_builder
        cls.vmi11_builder = ForestBuilderTestBench.vmi11_builder
        cls.vmi12_builder = ForestBuilderTestBench.vmi12_builder
        cls.vmi13_builder = ForestBuilderTestBench.vmi13_builder

        cls.vmi9_stands = ForestBuilderTestBench.vmi9_built()
        cls.vmi10_stands = ForestBuilderTestBench.vmi10_built()
        cls.vmi11_stands = ForestBuilderTestBench.vmi11_built()
        cls.vmi12_stands = ForestBuilderTestBench.vmi12_built()
        cls.vmi13_stands = ForestBuilderTestBench.vmi13_built()

        cls.vmi9_stands_ref_trees_false = ForestBuilderTestBench.vmi9_built({'measured_trees': False, 'strata': True})
        cls.vmi10_stands_ref_trees_false = ForestBuilderTestBench.vmi10_built({'measured_trees': False, 'strata': True})
        cls.vmi11_stands_ref_trees_false = ForestBuilderTestBench.vmi11_built({'measured_trees': False, 'strata': True})
        cls.vmi12_stands_ref_trees_false = ForestBuilderTestBench.vmi12_built({'measured_trees': False, 'strata': True})
        cls.vmi13_stands_ref_trees_false = ForestBuilderTestBench.vmi13_built({'measured_trees': False, 'strata': True})
