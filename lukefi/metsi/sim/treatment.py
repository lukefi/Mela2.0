from functools import partial
from typing import Callable, Optional
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.collected_data import OpTuple


type TreatmentFn[T: ComputationalUnit] = Callable[[T], OpTuple[T]]


class PreparedTreatment[T: ComputationalUnit]:
    name: str
    treatment_fn: TreatmentFn[T]
    tags: set[str]

    def __init__(self, treatment_fn: TreatmentFn[T], tags: Optional[set[str]] = None, **treatment_params):
        if tags is None:
            self.tags = set()
        else:
            self.tags = tags
        self.treatment_fn = partial(treatment_fn, **treatment_params)
        self.name = treatment_fn.__name__

    def __call__(self, state: T) -> OpTuple[T]:
        return self.treatment_fn(state)
