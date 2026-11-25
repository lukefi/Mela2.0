from collections.abc import Callable
from typing import Optional, Sequence

from lukefi.metsi.core.collected_data import CollectedData
from lukefi.metsi.core.model import ComputationalUnit
from lukefi.metsi.core.simulation_payload import SimulationPayload


type Predicate[T] = Callable[[T], bool]
type PostPredicate[T] = Callable[[T, Sequence[CollectedData]], bool]

class Condition[T: ComputationalUnit]:
    """
    Generic class which determine if condition of simulation payload apply to defined predicate.
    """
    __slots__ = ("predicate", "name", "time_points", "relative_time_points")

    predicate: Predicate[SimulationPayload[T]]
    """ Predicate which operates with given simulation payload.
    
    Return: boolean value
    """
    name: str
    """ (Optional) Name of the condition  """
    time_points: set[int]
    """ (Optional) Set of time points used by the predicate  """
    relative_time_points: set[int]
    """ (Optional) Set of relative time points used by the predicate  """

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

    def __invert__(self) -> "Condition[T]":
        return Condition(lambda x: not self.predicate(x), time_points=self.time_points)


class PostCondition[T:ComputationalUnit]:
    """
    Base class for predicate used to evaluate the applicability
    of both simulation payload and collected data.
    """
    __slots__ = ("predicate", "name")

    predicate: PostPredicate[SimulationPayload[T]]
    """ Post predicate evalutes variables from both simulation payload and collected data.
    
    Return: boolean value
    """
    name: str
    """ (Optional) Name of the condition """

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

    def __call__(self, payload: SimulationPayload[T], cd: Sequence[CollectedData]) -> bool:
        return self.predicate(payload, cd)

    def __and__(self, other: "PostCondition[T]") -> "PostCondition[T]":
        return PostCondition(lambda payload, cd: self.predicate(payload, cd) and other.predicate(payload, cd))

    def __or__(self, other: "PostCondition[T]") -> "PostCondition[T]":
        return PostCondition(lambda payload, cd: self.predicate(payload, cd) or other.predicate(payload, cd))
