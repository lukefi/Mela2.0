from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")
Predicate = Callable[[T], bool]


class Condition[T]:
    predicate: Predicate[T]

    def __init__(self, predicate: Predicate[T]) -> None:
        self.predicate = predicate

    def __call__(self, subject: T) -> bool:
        return self.predicate(subject)

    def __and__(self, other: "Condition[T]") -> "Condition[T]":
        return Condition(lambda x: self.predicate(x) and other.predicate(x))

    def __or__(self, other: "Condition[T]") -> "Condition[T]":
        return Condition(lambda x: self.predicate(x) or other.predicate(x))
