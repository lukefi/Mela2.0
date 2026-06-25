import sqlite3
from typing import Generator, Optional
from typing import Sequence as Sequence_

from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.generators import Alternatives, EventGeneratorBase, EventGenerator, Sequence
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.transition import TransitionFn


class SimulationInstruction[T: ComputationalUnit]:
    conditions: Sequence_[Condition[T]]
    event_generator: EventGenerator[T]

    def __init__(self, events: EventGenerator[T] | list[EventGeneratorBase] | set[EventGeneratorBase],
                 conditions: Optional[Sequence_[Condition[T]]] = None) -> None:
        if isinstance(events, EventGenerator):
            self.event_generator = events
        elif isinstance(events, list):
            self.event_generator = Sequence(events)
        elif isinstance(events, set):
            self.event_generator = Alternatives(list(events))
        if conditions is not None:
            self.conditions = conditions
        else:
            self.conditions = []

    def time_points(self, start_time: int) -> set[int]:
        return set().union(*[condition.time_points |
                             set(map(lambda t: t + start_time,
                                     condition.relative_time_points)) for condition in self.conditions])

    def evaluate(self,
                 payload: SimulationPayload[T],
                 db: sqlite3.Connection | None = None,
                 node: int = 0) -> Generator[SimulationPayload[T]]:
        yield from self.event_generator.evaluate(payload, db, node)


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
