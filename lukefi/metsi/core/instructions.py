import sqlite3
from typing import Generator, Optional
from typing import Sequence as Sequence_

from lukefi.metsi.core.condition import Condition
from lukefi.metsi.core.generators import Alternatives, EventGenerator, EventGeneratorBase, Sequence
from lukefi.metsi.core.model import ComputationalUnit
from lukefi.metsi.core.simulation_payload import SimulationPayload



class SimulationInstruction[T: ComputationalUnit]:
    __slots__ = ("conditions", "events")

    conditions: Sequence_[Condition[T]]
    events: EventGenerator[T]

    def __init__(self, events: EventGenerator[T] | list[EventGeneratorBase] | set[EventGeneratorBase],
                 conditions: Optional[Sequence_[Condition[T]]] = None) -> None:
        if isinstance(events, EventGenerator):
            self.events = events
        elif isinstance(events, list):
            self.events = Sequence(events)
        elif isinstance(events, set):
            self.events = Alternatives(list(events))
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
        yield from self.events.evaluate(payload, db, node)
