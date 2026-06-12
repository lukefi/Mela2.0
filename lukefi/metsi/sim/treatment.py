from typing import Any, Callable, Generic, Mapping, Optional, TypeVar
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

    def __call__(self, unit: T_contra, **params) -> Any:
        return self.treatment_fn(unit, **params)


class PredeterminedTreatment(Generic[T_contra]):

    name: str
    treatment_fn: TreatmentFn[T_contra]
    tags: set[str]
    static_parameters: dict[str, Any]
    file_parameters: dict[str, str]
    dynamic_parameters: Mapping[str, Callable[[T_contra], Any]]
    evaluated_params: dict[str, Any]
    collected_data: CollectableDataTypes

    def __init__(self,
                 name: str,
                 treatment_fn: TreatmentFn[T_contra],
                 tags: set[str] | None = None,
                 static_parameters: dict[str, Any] | None = None,
                 file_parameters: dict[str, str] | None = None,
                 dynamic_parameters: Mapping[str, Callable[[T_contra], Any]] | None = None,
                 collected_data: CollectableDataTypes | None = None):
        self.name = name
        self.treatment_fn = treatment_fn

        if tags is None:
            self.tags = set()
        else:
            self.tags = tags

        if static_parameters is None:
            self.static_parameters = {}
        else:
            self.static_parameters = static_parameters

        if file_parameters is None:
            self.file_parameters = {}
        else:
            self.file_parameters = file_parameters

        if dynamic_parameters is None:
            self.dynamic_parameters = {}
        else:
            self.dynamic_parameters = dynamic_parameters

        if collected_data is None:
            self.collected_data = set()
        else:
            self.collected_data = collected_data

    def __call__(self, unit: T_contra) -> OpTuple[T_contra]:
        return self.treatment_fn(unit, **self._evaluate_parameters(unit))

    def _evaluate_parameters(self, unit: T_contra) -> dict[str, Any]:
        # TODO: Check file parameter validity
        # TODO: Check parameter name collisions
        evaluated_dynamic = {name: expression(unit) for name, expression in self.dynamic_parameters.items()}
        self.evaluated_params = self.static_parameters | self.file_parameters | evaluated_dynamic
        return self.evaluated_params


do_nothing = Treatment[ComputationalUnit](do_nothing_, "do_nothing", {"nothing"})
