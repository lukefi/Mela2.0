from typing import Optional, TypeVar
from typing import Sequence as Sequence_

from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.event_tree import EventTree
from lukefi.metsi.sim.generators import Alternatives, EventGeneratorBase, EventGenerator, Sequence

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

    def unwrap(self) -> list[EventTree[T]]:
        return self.event_generator.compose_nested()
