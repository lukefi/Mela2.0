import csv
import os
import importlib.util
from collections.abc import Callable
from pathlib import Path
import sqlite3
from typing import Any
from lukefi.metsi.data.formats.forest_builder import VMI13Builder, VMI12Builder, XMLBuilder, GeoPackageBuilder
from lukefi.metsi.data.formats.io_utils import (
    stands_to_csv_content,
    csv_content_to_stands,
    stands_to_rst_content,
    mela_par_file_content)
from lukefi.metsi.app.app_io import MetsiConfiguration
from lukefi.metsi.app.app_types import ExportableContainer
from lukefi.metsi.domain.forestry_types import SimResults
from lukefi.metsi.domain.forestry_types import StandList, ForestStand
from lukefi.metsi.data.formats.declarative_conversion import Conversion
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.sim.collected_data import CollectedData

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

    if container_format == "csv":
        return csv_writer
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


def source_data_reader(state_format: str, conversions, **builder_flags) -> StandReader:
    """Resolve and prepare a reader function for non-FDM data formats"""
    if state_format == "vmi13":
        return lambda path: VMI13Builder(builder_flags, conversions.get('vmi13', {}), vmi_file_reader(path)).build()
    if state_format == "vmi12":
        return lambda path: VMI12Builder(builder_flags, conversions.get('vmi12', {}), vmi_file_reader(path)).build()
    if state_format == "xml":
        return lambda path: XMLBuilder(builder_flags, conversions.get('xml', {}), xml_file_reader(path)).build()
    if state_format == "gpkg":
        return lambda path: GeoPackageBuilder(builder_flags, conversions.get('gpkg', {}), str(path)).build()
    raise MetsiException(f"Unsupported state format '{state_format}'")


def read_stands_from_file(app_config: MetsiConfiguration, conversions: dict[str, Conversion]) -> StandList:
    """
    Read a list of ForestStands from given file with given configuration. Directly reads CSV format data. Utilizes
    FDM ForestBuilder utilities to transform VMI12, VMI13 or Forest Centre data into CSV ForestStand format.

    :param app_config: Mela2Configuration
    :return: list of ForestStands as computational units for simulation
    """
    if app_config.state_format == "csv":
        return csv_reader()(app_config.input_path)
    if app_config.state_format in ("vmi13", "vmi12", "xml", "gpkg"):
        return source_data_reader(
            app_config.state_format.value,
            conversions,
            strata=app_config.strata,
            measured_trees=app_config.measured_trees,
            strata_origin=app_config.strata_origin)(app_config.input_path)
    raise MetsiException(f"Unsupported state format '{app_config.state_format}'")


def read_control_module(control_path: str, control: str = "control_structure") -> dict[str, Any]:
    config_path = Path(control_path).resolve()  # Ensure absolute path
    module_name = config_path.stem  # Extract filename without extension

    spec = importlib.util.spec_from_file_location(module_name, str(control_path))
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, control):  # Check if variable exists
            return getattr(module, control)
        raise AttributeError(f"Variable '{control}' not found in {config_path}")
    raise ImportError(f"Could not load control module from {config_path}")


def row_writer(filepath: Path, rows: list[str]):
    with open(filepath, 'a', newline='\n', encoding="utf-8") as file:
        for row in rows:
            file.write(row)
            file.write('\n')


def csv_writer(filepath: Path, container: ExportableContainer[ForestStand]):
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


def vmi_file_reader(file: str | Path) -> list[str]:
    with open(file, 'r', encoding='utf-8') as input_file:
        return input_file.readlines()


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
    db = sqlite3.connect(file_path)
    return db
