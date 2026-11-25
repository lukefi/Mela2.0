from copy import deepcopy
from typing import Optional

from lukefi.metsi.core.model import ComputationalUnit, Finalizable

type OperationHistory = list[tuple[int, str, dict[str, dict], set[str]]]


class SimulationPayload[T: ComputationalUnit]:
    """
    Execution context associated with a node in the simulation.

    A payload accompanies a computational unit as it progresses through the
    simulation. It combines the current domain state with metadata
    required to track, evaluate and validate state within the simulation.
    """

    __slots__ = ("unit", "operation_history", "node_id")

    unit: T
    """ Computational unit representing the current state at this simulation node. """
    operation_history: OperationHistory
    """ Record of treatments performed on the unit along its current simulation path. """
    node_id: list[int]
    """ List of identifiers representing node order number in schedule """

    def __init__(self,
                 computational_unit: T,
                 operation_history: Optional[OperationHistory] = None,
                 node_id: Optional[list[int]] = None) -> None:
        self.unit = computational_unit

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
        if isinstance(self.unit, Finalizable):
            copy_like = self.unit.finalize()
        else:
            copy_like = deepcopy(self.unit)

        return SimulationPayload(
            computational_unit=copy_like,
            operation_history=list(self.operation_history),
            node_id=deepcopy(self.node_id)
        )
