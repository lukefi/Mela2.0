from copy import copy
from functools import partial
from typing import Callable, Generic, Optional, TypeVar
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.collected_data import CollectableDataTypes, OpTuple
from lukefi.metsi.sim.operations import do_nothing as do_nothing_

T_contra = TypeVar("T_contra", bound=ComputationalUnit, contravariant=True)
TreatmentFn = Callable[[T_contra], OpTuple[T_contra]]


class Treatment(Generic[T_contra]):
    """
    Class for wrapping a TreatmentFn with all necessary metadata.
    """

    name: str
    """
    A name unique to this Treatment. This will be used to identify the done treatment in the `nodes` table of the output
    database. Defaults to the `__name__` of the treatment function.
    """

    treatment_fn: TreatmentFn[T_contra]
    """
    The actual function that operates on the simulation state.
    """

    default_tags: set[str]
    """
    A set of tags that is always associated with this Treatment. This set will be combined with the Event specific tags
    when writing the simulation node to the output database.
    """

    collected_data: CollectableDataTypes
    """
    The set of different types of CollectedData that are returned by the treatment function.
    """

    def __init__(self,
                 treatment_fn: TreatmentFn[T_contra],
                 name: Optional[str] = None,
                 default_tags: Optional[set[str]] = None,
                 collected_data: Optional[CollectableDataTypes] = None,
                 ) -> None:
        """
        Creates a Treatment object with the given treatment function and metadata.

        :param treatment_fn: The treatment function to wrap
        :type treatment_fn: TreatmentFn[T_contra]
        :param name: A name for the Treatment. Defaults to the `__name__` of the function.
        :type name: Optional[str]
        :param default_tags: A set of tags to always associate with this Treatment
        :type default_tags: Optional[set[str]]
        :param collected_data: The set of CollectedData types that the treatment function can return
        :type collected_data: Optional[CollectableDataTypes]
        """
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
