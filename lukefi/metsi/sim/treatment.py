from copy import copy
from functools import partial
from typing import Callable, Generic, Optional, TypeVar
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.collected_data import CollectableDataTypes, OpTuple
from lukefi.metsi.sim.operations import do_nothing as do_nothing_

T_contra = TypeVar("T_contra", bound=ComputationalUnit, contravariant=True)
TreatmentFn = Callable[[T_contra], OpTuple[T_contra]]


class Treatment(Generic[T_contra]):
    name: str
    treatment_fn: TreatmentFn[T_contra]
    default_tags: set[str]
    collected_data: CollectableDataTypes

    def __init__(self,
                 treatment_fn: TreatmentFn[T_contra],
                 name: Optional[str] = None,
                 default_tags: Optional[set[str]] = None,
                 collected_data: Optional[CollectableDataTypes] = None,
                 ) -> None:
        self.treatment_fn = treatment_fn
        if default_tags is None:
            self.default_tags = set()
        else:
            self.default_tags = default_tags

        if collected_data is None:
            self.collected_data = set()
        else:
            self.collected_data = collected_data
        if name is None:
            self.name = treatment_fn.__name__
        else:
            self.name = name


class PreparedTreatment[T: ComputationalUnit]:
    name: str
    treatment_fn: TreatmentFn[T]
    tags: set[str]

    def __init__(self, treatment: Treatment[T], event_tags: Optional[set[str]] = None, **treatment_params):
        self.tags = copy(treatment.default_tags)
        if event_tags is not None:
            self.tags |= event_tags

        self.treatment_fn = partial(treatment.treatment_fn, **treatment_params)
        self.name = treatment.name

    def __call__(self, state: T) -> OpTuple[T]:
        return self.treatment_fn(state)

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


do_nothing = Treatment[ComputationalUnit](do_nothing_, "do_nothing", {"nothing"})
