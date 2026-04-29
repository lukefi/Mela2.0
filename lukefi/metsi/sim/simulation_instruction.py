import sqlite3
from typing import Generator, Optional, TypeVar
from typing import Sequence as Sequence_

from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.generators import Alternatives, EventGeneratorBase, EventGenerator, Sequence
from lukefi.metsi.sim.simulation_payload import SimulationPayload

T = TypeVar('T', bound=ComputationalUnit)  # T = ForestStand


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
                 node: int = 0) -> Generator[SimulationPayload[T], None, None]:
        yield from self.event_generator.evaluate(payload, db, node)
