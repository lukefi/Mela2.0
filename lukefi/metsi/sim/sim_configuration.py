from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.collected_data import CollectableDataTypes
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.transition import Transition, TransitionFn


class SimConfiguration[T: ComputationalUnit]:
    """
    A class to manage simulation configuration, including treatments, generators,
    events, and time points.
    Attributes:
        instructions: A list of instructions for the simulation.
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


class UpdatingInstructions[T: ComputationalUnit]:
    target_time: int
    transition: TransitionFn[T]

    output_transition_state: bool
    output_transition_cd: bool
    output_treatment_state: bool
    output_treatment_cd: bool

    def __init__(self,
                 target_time: int,
                 transition: TransitionFn[T],
                 output_transition_state: bool,
                 output_transition_cd: bool,
                 output_treatment_state: bool,
                 output_treatment_cd: bool) -> None:
        self.target_time = target_time
        self.transition = transition
        self.output_transition_state = output_transition_state
        self.output_transition_cd = output_transition_cd
        self.output_treatment_state = output_treatment_state
        self.output_treatment_cd = output_treatment_cd
