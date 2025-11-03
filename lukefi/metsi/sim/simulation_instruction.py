from typing import Optional, TypeVar

from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.generators import Alternatives, GeneratorBase, Generator, Sequence
from lukefi.metsi.sim.simulation_payload import SimulationPayload

T = TypeVar('T', bound=ComputationalUnit)  # T = ForestStand


class SimulationInstruction[T: ComputationalUnit]:
    conditions: list[Condition[SimulationPayload[T]]]
    event_generator: Generator[T]

    def __init__(self, time_points: list[int], events: Generator[T] | list[GeneratorBase] | set[GeneratorBase],
                 conditions: Optional[list[Condition[SimulationPayload[T]]]] = None) -> None:
        self.time_points = time_points
        if isinstance(events, Generator):
            self.event_generator = events
        elif isinstance(events, list):
            self.event_generator = Sequence(events)
        elif isinstance(events, set):
            self.event_generator = Alternatives(list(events))
        if conditions is not None:
            self.conditions = conditions
        else:
            self.conditions = []

    def unwrap(self, payload: SimulationPayload[T]):
        yield from self.event_generator.compose_nested().evaluate(payload)
