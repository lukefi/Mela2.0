from abc import ABC, abstractmethod
import os
from typing import Any, Optional, TypeVar, override
from typing import Sequence as Sequence_

from collections.abc import Callable
from lukefi.metsi.domain.collected_data import CollectableData
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.operations import prepared_treatment
from lukefi.metsi.sim.processor import processor
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.event_tree import EventTree
from lukefi.metsi.sim.simulation_payload import SimulationPayload, ProcessedTreatment
from lukefi.metsi.app.utils import MetsiException

T = TypeVar("T", bound=ComputationalUnit)

GeneratorFn = Callable[[Optional[list[EventTree[T]]], ProcessedTreatment[T]], list[EventTree[T]]]
TreatmentFn = Callable[[T], OpTuple[T]]
ProcessedGenerator = Callable[[Optional[list[EventTree[T]]]], list[EventTree[T]]]


class GeneratorBase[T: ComputationalUnit](ABC):
    """Shared abstract base class for Generator and Event types."""
    @abstractmethod
    def unwrap(self, parents: list[EventTree[T]], time_point: int) -> list[EventTree[T]]:
        pass

    @abstractmethod
    def get_types_of_collected_data(self) -> set[CollectableData]:
        pass


class Generator[T: ComputationalUnit](GeneratorBase, ABC):
    """Abstract base class for generator types."""
    children: Sequence_[GeneratorBase]
    time_point: Optional[int]

    def __init__(self, children: Sequence_[GeneratorBase], time_point: Optional[int] = None):
        self.children = children
        self.time_point = time_point

    def compose_nested(self) -> EventTree[T]:
        """
        Generate a simulation EventTree using the given NestableGenerator.

        :param nestable_generator: NestableGenerator tree for generating a EventTree.
        :return: The root node of the generated EventTree
        """
        root: EventTree[T] = EventTree()
        self.unwrap([root], 0)
        return root

    @override
    def get_types_of_collected_data(self) -> set[CollectableData]:
        retval = set()
        for child in self.children:
            retval.update(child.get_types_of_collected_data())
        return retval


class Sequence[T: ComputationalUnit](Generator[T]):
    """Generator for sequential events."""

    @override
    def unwrap(self, parents: list[EventTree], time_point: int) -> list[EventTree]:
        current = parents
        for child in self.children:
            current = child.unwrap(current, self.time_point or time_point)
        return current


class Alternatives[T: ComputationalUnit](Generator[T]):
    """Generator for branching events"""

    @override
    def unwrap(self, parents: list[EventTree], time_point: int) -> list[EventTree]:
        retval = []
        for child in self.children:
            retval.extend(child.unwrap(parents, self.time_point or time_point))
        return retval


class Event[T: ComputationalUnit](GeneratorBase):
    """Base class for events. Contains conditions and parameters and the actual treatment function that operates on the
    simulation state."""
    treatment: TreatmentFn[T]
    parameters: dict[str, Any]
    file_parameters: dict[str, str]
    preconditions: list[Condition[SimulationPayload[T]]]
    postconditions: list[Condition[SimulationPayload[T]]]
    tags: set[str]
    collected_data: set[CollectableData]

    def __init__(self, treatment: TreatmentFn[T], parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[Condition[SimulationPayload[T]]]] = None,
                 postconditions: Optional[list[Condition[SimulationPayload[T]]]] = None,
                 file_parameters: Optional[dict[str, str]] = None,
                 tags: Optional[set[str]] = None,
                 collected_data: Optional[set[CollectableData]] = None) -> None:
        self.treatment = treatment

        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = {}

        if file_parameters is not None:
            self.file_parameters = file_parameters
        else:
            self.file_parameters = {}

        if preconditions is not None:
            self.preconditions = preconditions
        else:
            self.preconditions = []

        if postconditions is not None:
            self.postconditions = postconditions
        else:
            self.postconditions = []

        if tags is not None:
            self.tags = tags
        else:
            self.tags = set()

        if collected_data is not None:
            self.collected_data = collected_data
        else:
            self.collected_data = set()

    @override
    def unwrap(self, parents: list[EventTree], time_point: int) -> list[EventTree]:
        retval = []
        for parent in parents:
            branch = EventTree(self._prepare_paremeterized_treatment(time_point))
            parent.add_branch(branch)
            retval.append(branch)
        return retval

    @override
    def get_types_of_collected_data(self) -> set[CollectableData]:
        return self.collected_data

    def _prepare_paremeterized_treatment(self, time_point) -> ProcessedTreatment[T]:
        self._check_file_params()
        combined_params = self._merge_params()
        treatment = prepared_treatment(self.treatment, **combined_params)
        return lambda payload: processor(payload, treatment, self.treatment, time_point,
                                         self.preconditions, self.postconditions, **combined_params)

    def _check_file_params(self):
        for _, path in self.file_parameters.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(f"file {path} defined in operation_file_params was not found")

    def _merge_params(self) -> dict[str, Any]:
        common_keys = self.parameters.keys() & self.file_parameters.keys()
        if common_keys:
            raise MetsiException(
                f"parameter(s) {common_keys} were defined both in 'parameters' and 'file_parameters' sections "
                "in control.py. Please change the name of one of them.")
        return self.parameters | self.file_parameters  # pipe is the merge operator
