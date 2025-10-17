from collections.abc import Callable
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypeVar

from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.collected_data import CollectedData
from lukefi.metsi.sim.finalizable import Finalizable
if TYPE_CHECKING:
    from lukefi.metsi.sim.generators import TreatmentFn


class SimulationPayload[T: ComputationalUnit](SimpleNamespace):
    """Data structure for keeping simulation state and progress data. Passed on as the data package of chained
    operation calls. """
    computational_unit: T
    operation_history: list[tuple[int, "TreatmentFn[T]", dict[str, dict]]]

    def __copy__(self) -> "SimulationPayload[T]":
        copy_like: T
        if isinstance(self.computational_unit, Finalizable):
            copy_like = self.computational_unit.finalize()
        else:
            copy_like = deepcopy(self.computational_unit)

        return SimulationPayload(
            computational_unit=copy_like,
            operation_history=list(self.operation_history)
        )

T = TypeVar("T", bound=ComputationalUnit)
ProcessedTreatment = Callable[[SimulationPayload[T]], tuple[SimulationPayload[T], list[CollectedData]]]
