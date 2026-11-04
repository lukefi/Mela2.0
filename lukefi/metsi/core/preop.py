from dataclasses import dataclass
from typing import Any, Callable, Sequence

from lukefi.metsi.core.model import ComputationalUnit


type PreprocessingFn[T: ComputationalUnit] = Callable[[Sequence[T]], Sequence[T]]


@dataclass(init=False, slots=True)
class PreprocessingOperation[T: ComputationalUnit]:
    operation: PreprocessingFn[T]
    params: dict[str, Any]

    def __init__(self, operation: PreprocessingFn[T], **params):
        self.operation = operation
        self.params = params

    def __call__(self, units: Sequence[T]) -> Sequence[T]:
        return self.operation(units, **self.params)
