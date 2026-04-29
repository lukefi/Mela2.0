from collections.abc import Callable
import sqlite3
from typing import Any, Optional
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.domain.utils.file_io import output_node_to_db
from lukefi.metsi.sim.collected_data import CollectableDataTypes, OpTuple
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.simulation_payload import OperationHistory, SimulationPayload

type TransitionFn[T: ComputationalUnit] = Callable[[T, int], OpTuple[T]]
type TransitionInitFn[T: ComputationalUnit] = Callable[[T, dict[str, Any]], None]


class Transition[T: ComputationalUnit]:
    transition_fn: TransitionFn[T]
    parameters: dict[str, Any]
    init_fn: TransitionInitFn[T] | None
    name: str
    collected_data: CollectableDataTypes
    db_output: bool
    db_output_state: bool
    db_output_cd: bool
    max_step: int

    def __init__(
        self,
        transition_fn: TransitionFn[T],
        max_step: int = 5,
        collected_data: Optional[CollectableDataTypes] = None,
        name: Optional[str] = None,
        db_output: bool = True,
        db_output_state: bool = False,
        db_output_cd: bool = True,
        *,
        init_fn: TransitionInitFn[T] | None = None,
        **parameters,
    ):
        self.transition_fn = transition_fn
        self.max_step = max_step
        self.parameters = parameters
        self.init_fn = init_fn
        self.db_output = db_output
        self.db_output_state = db_output_state
        self.db_output_cd = db_output_cd

        if name is not None:
            self.name = name
        else:
            self.name = transition_fn.__name__

        if collected_data is not None:
            self.collected_data = collected_data
        else:
            self.collected_data = set()

    def initialize(self, unit: T) -> None:
        if self.init_fn is not None:
            self.init_fn(unit, self.parameters)

    def __call__(self,
                 payload: SimulationPayload[T],
                 db: Optional[sqlite3.Connection],
                 time_step: int,
                 transition_count: int = 1) -> OpTuple[T]:
        if self.max_step is not None:
            time_step = min(time_step, self.max_step)
        new_state, collected_data = self.transition_fn(payload.computational_unit, time_step, **self.parameters)

        if db is not None and self.db_output:
            temp_history: OperationHistory = [
                (payload.computational_unit.time, self.name, self.parameters, set())
            ]
            transition_payload = SimulationPayload(new_state, temp_history, payload.node_id)
            output_node_to_db(db, transition_payload, collected_data, output_state=self.db_output_state,
                              output_collected_data=self.db_output_cd, transition_count=transition_count)

        return new_state, collected_data

    def __str__(self) -> str:
        return f"{{transition_fn: {self.transition_fn.__name__}, " \
               f"max_step: {self.max_step}, parameters: {self.parameters}}}"

    def __repr__(self) -> str:
        return f"{{transition_fn: {self.transition_fn.__name__}, " \
               f"max_step: {self.max_step}, parameters: {self.parameters}}}"


class SimConfiguration[T: ComputationalUnit]:
    """
    A class to manage simulation configuration, including treatments, generators,
    events, and time points.
    Attributes:
        instructions: A list of instructions for the simulation.
        time_points: A sorted list of unique time points derived from the simulation instructions.
        collected_data: Set of CollectableData values describing the types of extra data collected by the simulation.
    Methods:
        __init__(**kwargs):
            Initializes the SimConfiguration instance with keyword arguments.
    """
    instructions: list[SimulationInstruction[T]] = []
    transition: Transition[T]
    end_condition: Condition[T]
    collected_data: CollectableDataTypes

    def __init__(self,
                 simulation_instructions: list[SimulationInstruction[T]],
                 transition: Transition[T],
                 end_condition: Condition[T]):
        """
        Initializes the core simulation object.

        :param simulation_instructions: list of SimulationInstruction declarations describing the structure of Events,
        Treatments and Conditions in the simulation run
        :type simulation_instructions: list[SimulationInstruction[T]]
        :param transition: the Transition used to unconditionally evolve the simulation state between evaluated
        SimulationInstructions
        :type transition: Transition[T]
        :param end_condition: Condition for ending the simulation (per branch)
        :type end_condition: Condition[T]
        """
        self.transition = transition
        self.instructions = simulation_instructions
        self.end_condition = end_condition
        self._get_collected_data_types()

    def _get_collected_data_types(self):
        collected_data = self.transition.collected_data
        for instruction in self.instructions:
            collected_data.update(instruction.event_generator.get_types_of_collected_data())
        self.collected_data = collected_data
