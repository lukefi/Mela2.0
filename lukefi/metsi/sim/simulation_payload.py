from copy import deepcopy
from typing import Optional
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.finalizable import Finalizable


type OperationHistory = list[tuple[int, str, dict[str, dict], set[str]]]

class SimulationPayload[T: ComputationalUnit]:
    """Data structure for keeping simulation state and progress data. Passed on as the data package of chained
    operation calls. """
    computational_unit: T
    operation_history: OperationHistory
    node_id: list[int]

    def __init__(self,
                 computational_unit: T,
                 operation_history: Optional[OperationHistory] = None,
                 node_id: Optional[list[int]] = None) -> None:
        self.computational_unit = computational_unit

        if operation_history is None:
            self.operation_history = []
        else:
            self.operation_history = operation_history

        if node_id is None:
            self.node_id = [0]
        else:
            self.node_id = node_id

    def __copy__(self) -> "SimulationPayload[T]":
        copy_like: T
        if isinstance(self.computational_unit, Finalizable):
            copy_like = self.computational_unit.finalize()
        else:
            copy_like = deepcopy(self.computational_unit)

        return SimulationPayload(
            computational_unit=copy_like,
            operation_history=list(self.operation_history),
            node_id=deepcopy(self.node_id)
        )
