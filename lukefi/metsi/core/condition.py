from collections.abc import Callable
from typing import Optional, Sequence

from lukefi.metsi.core.collected_data import CollectedData
from lukefi.metsi.core.model import ComputationalUnit
from lukefi.metsi.core.simulation_payload import SimulationPayload


type Predicate[T] = Callable[[T], bool]
type PostPredicate[T] = Callable[[T, Sequence[CollectedData]], bool]

class Condition[T: ComputationalUnit]:
    __slots__ = ("predicate", "name", "time_points", "relative_time_points")

    predicate: Predicate[SimulationPayload[T]]
    name: str
    time_points: set[int]
    relative_time_points: set[int]

    def __init__(self,
                 predicate: Predicate[SimulationPayload[T]],
                 name: Optional[str] = None,
                 time_points: Optional[set[int]] = None,
                 relative_time_points: Optional[set[int]] = None) -> None:
        self.predicate = predicate

        if name is None:
            self.name = predicate.__name__
        else:
            self.name = name

        if time_points is not None:
            self.time_points = time_points
        else:
            self.time_points = set()

        if relative_time_points is not None:
            self.relative_time_points = relative_time_points
        else:
            self.relative_time_points = set()

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def __call__(self, subject: SimulationPayload[T]) -> bool:
        return self.predicate(subject)

    def __and__(self, other: "Condition[T]") -> "Condition[T]":
        return Condition(lambda x: self.predicate(x) and other.predicate(x),
                         time_points=self.time_points | other.time_points)

    def __or__(self, other: "Condition[T]") -> "Condition[T]":
        return Condition(lambda x: self.predicate(x) or other.predicate(x),
                         time_points=self.time_points | other.time_points)


class PostCondition[T:ComputationalUnit]:
    __slots__ = ("predicate", "name")

    predicate: PostPredicate[SimulationPayload[T]]
    name: str

    def __init__(self,
                 predicate: PostPredicate[SimulationPayload[T]],
                 name: str | None = None):
        self.predicate = predicate
        if name is None:
            self.name = predicate.__name__
        else:
            self.name = name

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def __call__(self, unit: SimulationPayload[T], cd: Sequence[CollectedData]) -> bool:
        return self.predicate(unit, cd)

    def __and__(self, other: "PostCondition[T]") -> "PostCondition[T]":
        return PostCondition(lambda unit, cd: self.predicate(unit, cd) and other.predicate(unit, cd))

    def __or__(self, other: "PostCondition[T]") -> "PostCondition[T]":
        return PostCondition(lambda unit, cd: self.predicate(unit, cd) or other.predicate(unit, cd))
