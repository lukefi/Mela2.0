from collections.abc import Callable
from functools import partial
from typing import Any
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.collected_data import CollectableDataTypes, OpTuple
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction

type TransitionFn[T: ComputationalUnit] = Callable[[T], OpTuple[T]]


class Transition[T: ComputationalUnit]:
    transition_fn: TransitionFn[T]
    parameters: dict[str, Any]

    def __init__(self, transition_fn: TransitionFn[T], **parameters):
        self.parameters = parameters
        self.transition_fn = partial(transition_fn, **parameters)

    def __call__(self, state: T) -> OpTuple[T]:
        return self.transition_fn(state)


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
        self._get_collected_data_types(self.instructions)

    def _get_collected_data_types(self, instructions: list["SimulationInstruction[T]"]):
        collected_data = set()
        for instruction in instructions:
            collected_data.update(instruction.event_generator.get_types_of_collected_data())
        self.collected_data = collected_data
