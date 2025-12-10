from collections.abc import Callable
from typing import TypeVar

from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.simulation_payload import SimulationPayload


T = TypeVar("T")
Predicate = Callable[[T], bool]


class Condition[T: ComputationalUnit]:
    predicate: Predicate[SimulationPayload[T]]

    def __init__(self, predicate: Predicate[SimulationPayload[T]]) -> None:
        self.predicate = predicate

    def __call__(self, subject: SimulationPayload[T]) -> bool:
        return self.predicate(subject)

    def __and__(self, other: "Condition[T]") -> "Condition[T]":
        return Condition(lambda x: self.predicate(x) and other.predicate(x))

    def __or__(self, other: "Condition[T]") -> "Condition[T]":
        return Condition(lambda x: self.predicate(x) or other.predicate(x))
