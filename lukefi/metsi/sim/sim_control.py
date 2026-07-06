from dataclasses import dataclass
from typing import Any, Callable, Sequence

from lukefi.metsi.sim.collected_data import CollectableDataTypes
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.instructions import SimulationInstruction
from lukefi.metsi.sim.model import ComputationalUnit
from lukefi.metsi.sim.transition import Transition
from lukefi.metsi.sim.treatment import TreatmentFn


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
