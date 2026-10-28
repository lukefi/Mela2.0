from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Sequence

from lukefi.metsi.core.model import ComputationalUnit


type PreprocessingFn[T: ComputationalUnit] = Callable[[Sequence[T]], Sequence[T]]


@dataclass(init=False, slots=True)
class PreprocessingOperation[T: ComputationalUnit]:
    operation: PreprocessingFn[T]
    params: dict[str, Any]

    def __init__(self, operation: PreprocessingFn[T], params: dict[str, Any] | None = None):
        self.operation = operation
        if params is not None:
            self.params = params
        else:
            self.params = {}

    def __call__(self, units: Sequence[T]) -> Sequence[T]:
        return self.operation(units, **self.params)


def pre_op[T: ComputationalUnit](pre_op_fn: PreprocessingFn[T]):
    @wraps(pre_op_fn)
    def wrapper(**params):
        return PreprocessingOperation(pre_op_fn, params)
    return wrapper
