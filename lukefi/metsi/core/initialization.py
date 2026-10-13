
from typing import Any, Callable

from lukefi.metsi.core.model import ComputationalUnit


type InitFn[T: ComputationalUnit] = Callable[[T], None]


class Initialization[T: ComputationalUnit]:
    init_fn: InitFn[T]
    params: dict[str, Any]

    def __init__(self, init_fn: InitFn[T], init_params: dict[str, Any] | None = None) -> None:
        self.init_fn = init_fn
        self.params = init_params or {}

    def __call__(self, unit: T):
        self.init_fn(unit, **self.params)
