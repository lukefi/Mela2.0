from dataclasses import dataclass
from typing import Any

from lukefi.metsi.app.metsi_enum import RunMode, StateFormat, StrataOrigin
from lukefi.metsi.data.formats.declarative_conversion import Conversion
from lukefi.metsi.sim.computational_unit import ComputationalUnit
from lukefi.metsi.sim.sim_control import Preprocessing, Resimulation, Simulation, Updating
from lukefi.metsi.sim.utils import ConfigurationException


@dataclass
class AppConfiguration:
    input_path = ""
    target_directory = ""
    slice_percentage: float | None
    slice_size: int | None
    state_format: StateFormat
    run_modes: list[RunMode]
    preprocessing_output_file: str
    simulation_output_file: str
    sqlite_decl: dict[str, list[str]] | None
    measured_trees: bool
    strata: bool
    strata_origin: StrataOrigin

    def __init__(self,
                 *,
                 input_path: str = "",
                 target_directory: str = "",
                 slice_percentage: float | None = None,
                 slice_size: int | None = None,
                 state_format: StateFormat,
                 run_modes: list[RunMode],
                 preprocessing_output_file: str = "preprocessing_result",
                 simulation_output_file: str = "simulation_results",
                 sqlite_decl: dict[str, list[str]] | None = None,
                 measured_trees: bool = False,
                 strata: bool = True,
                 strata_origin: StrataOrigin = StrataOrigin.INVENTORY) -> None:
        self.input_path = input_path
        self.target_directory = target_directory
        self.slice_percentage = slice_percentage
        self.slice_size = slice_size
        self.state_format = state_format
        self.run_modes = run_modes
        self.preprocessing_output_file = preprocessing_output_file
        self.simulation_output_file = simulation_output_file
        self.sqlite_decl = sqlite_decl
        self.measured_trees = measured_trees
        self.strata = strata
        self.strata_origin = strata_origin

        self._validate_run_modes()

    def _validate_run_modes(self):
        if RunMode.EXPORT_PREPRO in self.run_modes and (
                RunMode.PREPROCESS not in self.run_modes and RunMode.UPDATE not in self.run_modes):
            raise ConfigurationException("Run mode EXPORT_PREPRO cannot be without PREPROCESS or UPDATE")


@dataclass
class MetsiControl[T: ComputationalUnit]:
    app_configuration: AppConfiguration
    conversions: dict[str, dict[str, Conversion]] | None = None
    preprocessing: Preprocessing[T] | None = None
    updating: Updating | None = None
    export_prepro: dict[str, dict[str, Any]] | None = None
    simulation: Simulation[T] | None = None
    resimulation: Resimulation[T] | None = None
