import csv
import os
import importlib.util
from collections.abc import Callable
from pathlib import Path
import sqlite3
from typing import Generator, Optional
from lukefi.metsi.data.formats.forest_centre.forest_centre_builder import GeoPackageBuilder, XMLBuilder
from lukefi.metsi.data.formats.nfi.vmi10_builder import VMI10Builder
from lukefi.metsi.data.formats.nfi.vmi11_builder import VMI11Builder
from lukefi.metsi.data.formats.nfi.vmi12_builder import VMI12Builder
from lukefi.metsi.data.formats.nfi.vmi13_builder import VMI13Builder

from lukefi.metsi.data.formats.io_utils import (
    stands_to_csv_content,
    csv_content_to_stands,
    csv_exp_content_to_stands,
    stands_to_rst_content,
    mela_par_file_content)
from lukefi.metsi.app.app_types import ExportableContainer
from lukefi.metsi.data.formats.nfi.vmi9_builder import VMI9Builder
from lukefi.metsi.domain.forestry_types import SimResults
from lukefi.metsi.domain.forestry_types import StandList, ForestStand
from lukefi.metsi.data.formats.declarative_conversion import Conversion
from lukefi.metsi.app.metsi_control import AppConfiguration, MetsiControl
from lukefi.metsi.sim.collected_data import CollectedData
from lukefi.metsi.data.util.csv_utils import STAND_INTERNAL_COLUMNS, csv_cell
from lukefi.metsi.data.vector_model import DTYPES_TREE, DTYPES_STRATA
from lukefi.metsi.data.model import stand_as_internal_row
from lukefi.metsi.sim.exceptions import MetsiException

StandReader = Callable[[str | Path], StandList]
StandWriter = Callable[[Path, ExportableContainer[ForestStand]], None]
ObjectLike = StandList | SimResults | CollectedData
ObjectWriter = Callable[[Path, ObjectLike], None]


def prepare_target_directory(path_descriptor: str) -> Path:
    """
    Sanity check a given directory path. Existing directory must be accessible for writing. Raise exception if directory
    is not usable. Create the directory if not existing.
    necessary.

    :param path_descriptor: relative directory path
    :return: Path instance for directory
    """
    if os.path.exists(path_descriptor):
        if os.path.isdir(path_descriptor) and os.access(path_descriptor, os.W_OK):
            return Path(path_descriptor)
        raise MetsiException(
            f"Output directory {path_descriptor} not available. Ensure it is a writable and empty, "
            "or a non-existing directory.")

    os.makedirs(path_descriptor)
    return Path(path_descriptor)


def stand_writer(container_format: str) -> StandWriter:
    """Return a serialization file writer function for a ForestDataPackage"""

    if container_format in ("csv", "csv_legacy"):
        return csv_legacy_writer
    if container_format == "csv_exp":
        return csv_exp_writer
    if container_format == "rst":
        return rst_writer
    raise MetsiException(f"Unsupported container format '{container_format}'")


def write_stands_to_file(
        result: ExportableContainer[ForestStand], filepath: Path, state_output_container: str):
    """Resolve a writer function for ForestStands matching the given state_output_container. Invokes write."""
    writer = stand_writer(state_output_container)
    writer(filepath, result)


def determine_file_path(dir_: str | Path, filename: str) -> Path:
    return Path(dir_, filename)


def csv_reader() -> StandReader:
    """Reads FDM data from CSV to SOA vectors"""

    return lambda path: csv_content_to_stands(csv_file_reader(path))


def csv_exp_reader() -> StandReader:
    """Reads FDM data from exp_csv directory to SOA vectors"""

    return csv_exp_content_to_stands


def source_data_reader(state_format: str, conversions, **builder_flags) -> StandReader:
    """Resolve and prepare a reader function for non-FDM data formats"""
    if state_format == "vmi13":
        return lambda path: VMI13Builder(builder_flags, conversions.get('vmi13', {}), vmi_file_reader(path)).build()
    if state_format == "vmi12":
        return lambda path: VMI12Builder(builder_flags, conversions.get('vmi12', {}), vmi_file_reader(path)).build()
    if state_format == "vmi11":
        return lambda path: VMI11Builder(builder_flags, conversions.get('vmi11', {}), vmi_file_reader(path)).build()
    if state_format == "vmi10":
        return lambda path: VMI10Builder(builder_flags, conversions.get('vmi10', {}), vmi_file_reader(path)).build()
    if state_format == "vmi9":
        return lambda path: VMI9Builder(builder_flags, conversions.get('vmi9', {}), vmi_file_reader(path)).build()
    if state_format == "xml":
        return lambda path: XMLBuilder(builder_flags, conversions.get('xml', {}), xml_file_reader(path)).build()
    if state_format == "gpkg":
        return lambda path: GeoPackageBuilder(builder_flags, conversions.get('gpkg', {}), str(path)).build()
    raise MetsiException(f"Unsupported state format '{state_format}'")


def read_stands_from_file(app_config: AppConfiguration,
                          conversions: dict[str, dict[str, Conversion]] | None = None) -> StandList:
    """
    Read a list of ForestStands from given file with given configuration. Directly reads CSV format data. Utilizes
    FDM ForestBuilder utilities to transform VMI12, VMI13 or Forest Centre data into CSV ForestStand format.

    :param app_config: Mela2Configuration
    :return: list of ForestStands as computational units for simulation
    """
    if app_config.state_format == "csv":
        return csv_reader()(app_config.input_path)
    if app_config.state_format == "csv_exp":
        return csv_exp_reader()(app_config.input_path)
    if app_config.state_format in ("vmi13", "vmi12", "vmi11", "vmi10", "vmi9", "xml", "gpkg"):
        return source_data_reader(
            app_config.state_format.value,
            conversions,
            strata=app_config.strata,
            measured_trees=app_config.measured_trees,
            strata_origin=app_config.strata_origin)(app_config.input_path)
    raise MetsiException(f"Unsupported state format '{app_config.state_format}'")


def read_control_module(control_path: str, cli_arguments: dict, control: str = "control_structure") -> MetsiControl:
    config_path = Path(control_path).resolve()  # Ensure absolute path
    module_name = config_path.stem  # Extract filename without extension

    spec = importlib.util.spec_from_file_location(module_name, str(control_path))
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, control):  # Check if variable exists
            retval: MetsiControl = getattr(module, control)
            cfg = retval.app_configuration
            if "input_path" in cli_arguments:
                cfg.input_path = cli_arguments["input_path"]
            if "target_directory" in cli_arguments:
                cfg.target_directory = cli_arguments["target_directory"]
            return retval
        raise AttributeError(f"Variable '{control}' not found in {config_path}")
    raise ImportError(f"Could not load control module from {config_path}")


def row_writer(filepath: Path, rows: list[str]):
    with open(filepath, 'a', newline='\n', encoding="utf-8") as file:
        for row in rows:
            file.write(row)
            file.write('\n')


def csv_legacy_writer(filepath: Path, container: ExportableContainer[ForestStand]):
    row_writer(filepath, stands_to_csv_content(container, ';'))


def rst_writer(filepath: Path, container: ExportableContainer[ForestStand]):
    rows = stands_to_rst_content(container)
    row_writer(filepath, rows)
    if container.additional_vars is not None:
        par_writer(filepath, container.additional_vars)


def par_writer(filepath: Path, var_names: list[str]):
    def to_par_filepath(filepath: Path):
        dir_parts = list(filepath.parts)[0:-1]
        return determine_file_path(os.path.join(*dir_parts), 'c-variables.par')
    row_writer(to_par_filepath(filepath), mela_par_file_content(var_names))


def csv_exp_writer(filepath: Path, container: ExportableContainer[ForestStand]) -> None:
    """
    Exports stands.csv, trees.csv and strata.csv
    """
    out_dir = filepath.parent

    stands_path = out_dir / "stands.csv"
    trees_path = out_dir / "trees.csv"
    strata_path = out_dir / "strata.csv"

    additional = container.additional_vars or []

    # --- stands.csv ---
    stand_header = ["stand_identifier"] + STAND_INTERNAL_COLUMNS + list(additional)
    with open(stands_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(stand_header)
        for stand in container.export_objects:
            row = [stand.identifier] + stand_as_internal_row(stand)
            if additional:
                row.extend(stand.get_value_list(additional))
            w.writerow([csv_cell(x) for x in row])

    # --- trees.csv ---
    tree_cols = list(DTYPES_TREE.keys())
    tree_header = ["stand_identifier"] + tree_cols
    with open(trees_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(tree_header)
        for stand in container.export_objects:
            trees = stand.reference_trees
            for i in range(trees.size):
                row = [stand.identifier] + [getattr(trees, col)[i] for col in tree_cols]
                w.writerow([csv_cell(x) for x in row])

    # --- strata.csv ---
    strata_cols = list(DTYPES_STRATA.keys())
    strata_header = ["stand_identifier"] + strata_cols
    with open(strata_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(strata_header)
        for stand in container.export_objects:
            strata = stand.tree_strata
            for i in range(strata.size):
                row = [stand.identifier] + [getattr(strata, col)[i] for col in strata_cols]
                w.writerow([csv_cell(x) for x in row])


def vmi_file_reader(file: str | Path) -> Generator[str]:
    with open(file, 'r', encoding='utf-8') as input_file:
        yield from map(lambda row: row.strip("\n"), input_file)


def xml_file_reader(file: str | Path) -> str:
    with open(file, 'r', encoding='utf-8') as input_file:
        return input_file.read()


def csv_file_reader(file: str | Path) -> list[list[str]]:
    with open(file, 'r', encoding='utf-8') as input_file:
        return list(csv.reader(input_file, delimiter=';'))


def init_sqlite_database(file_path: str | Path) -> sqlite3.Connection:
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
        except OSError as e:
            raise MetsiException(f"Unable to delete existing database file {file_path}") from e
    db = sqlite3.connect(file_path, autocommit=False)
    return db


def delete_existing_export_files(
    target_directory: str,
    export_prepro: Optional[dict],
    preprocessing_base_name: str,
    simulation_base_name: str,
    force_delete: bool,
) -> bool:
    """
    Checks whether export_prepro output files or database already exist (csv/rst/db).
    If they exist:
      - if force_delete: delete and continue
      - else prompt user y/n
    Returns True if execution should continue, False if it should exit.
    """
    formats = []
    if export_prepro:
        formats = list(export_prepro.keys())

    td = Path(target_directory)

    candidates: list[Path] = []
    candidates.append(td / f"{simulation_base_name}.db")

    for fmt in formats:
        if fmt == "csv_exp":
            candidates.extend([
                td / "stands.csv",
                td / "trees.csv",
                td / "strata.csv",
                td / "metadata.json",
            ])
        else:
            candidates.append(td / f"{preprocessing_base_name}.{fmt}")

        if fmt == "rst":
            candidates.append(td / "c-variables.par")

    seen = set()
    unique_candidates: list[Path] = []
    for p in candidates:
        key = str(p.resolve()) if p.is_absolute() else str(p)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(p)

    existing = [p for p in unique_candidates if p.is_file()]
    if not existing:
        return True

    if not force_delete:
        print("Output file(s) already exist:")
        for p in existing:
            print(f"  - {p}")
        answer = input("Do you want to delete them and continue? (y/n): ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborting (no files were deleted).")
            return False

    for p in existing:
        try:
            p.unlink()
        except OSError as e:
            raise MetsiException(f"Unable to delete existing output file: {p}") from e

    return True
