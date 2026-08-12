from dataclasses import dataclass
from typing import Any, Callable, Sequence

from lukefi.metsi.app.metsi_enum import RunMode, StateFormat, StrataOrigin
from lukefi.metsi.app.utils import ConfigurationException
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.data.formats.declarative_conversion import Conversion
from lukefi.metsi.sim.collected_data import CollectableDataTypes
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.instructions import SimulationInstruction
from lukefi.metsi.sim.transition import Transition
from lukefi.metsi.sim.treatment import TreatmentFn


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


type PreprocessingOperation[T: ComputationalUnit] = Callable[[Sequence[T]], Sequence[T]]


@dataclass
class Preprocessing[T:ComputationalUnit]:
    operations: Sequence[PreprocessingOperation[T]]
    params: dict[PreprocessingOperation[T], list[dict[str, Any]]]


@dataclass
class Updating[T: ComputationalUnit]:
    target_time: int
    transition: Transition[T]

    output_treatment_state: bool
    output_treatment_cd: bool

    def __init__(self,
                 target_time: int,
                 transition: Transition[T],
                 *,
                 output_treatment_state: bool = True,
                 output_treatment_cd: bool = True) -> None:
        self.target_time = target_time
        self.transition = transition
        self.output_treatment_state = output_treatment_state
        self.output_treatment_cd = output_treatment_cd


@dataclass
class Resimulation[T: ComputationalUnit]:
    transition: Transition[T]
    selected_schedules_file: str
    treatment_map: dict[str, TreatmentFn[T]]
    collected_data: CollectableDataTypes
    output_treatment_state: bool
    output_treatment_cd: bool

    def __init__(self,
                 transition: Transition[T],
                 schedules_file: str,
                 *,
                 treatment_map: dict[str, TreatmentFn[T]] | None = None,
                 collected_data: CollectableDataTypes | None = None,
                 output_treatment_state: bool = True,
                 output_treatment_cd: bool = True) -> None:
        self.transition = transition
        self.schedules_file = schedules_file
        if treatment_map is None:
            self.treatment_map = {}
        else:
            self.treatment_map = treatment_map
        if collected_data is None:
            self.collected_data = set()
        else:
            self.collected_data = collected_data
        self.output_treatment_state = output_treatment_state
        self.output_treatment_cd = output_treatment_cd


@dataclass
class Simulation[T: ComputationalUnit]:
    instructions: Sequence[SimulationInstruction[T]]
    transition: Transition[T]
    end_condition: Condition[T]
    collected_data: CollectableDataTypes

    def __init__(self,
                 instructions: Sequence[SimulationInstruction[T]],
                 transition: Transition[T],
                 end_condition: Condition[T]):
        self.instructions = instructions
        self.transition = transition
        self.end_condition = end_condition
        self._determine_collected_data_types()

    def _determine_collected_data_types(self):
        collected_data = self.transition.collected_data
        for instruction in self.instructions:
            collected_data.update(instruction.event_generator.get_types_of_collected_data())
        self.collected_data = collected_data


@dataclass
class MetsiControl[T: ComputationalUnit]:
    app_configuration: AppConfiguration
    conversions: dict[str, dict[str, Conversion]] | None = None
    preprocessing: Preprocessing[T] | None = None
    updating: Updating | None = None
    export_prepro: dict[str, dict[str, Any]] | None = None
    simulation: Simulation[T] | None = None
    resimulation: Resimulation[T] | None = None
