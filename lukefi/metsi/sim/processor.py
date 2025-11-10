from copy import deepcopy
from typing import TYPE_CHECKING
from lukefi.metsi.app.utils import ConditionFailed
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.collected_data import CollectedData
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.simulation_payload import SimulationPayload
if TYPE_CHECKING:
    from lukefi.metsi.sim.generators import TreatmentFn


def processor[T: ComputationalUnit](payload: SimulationPayload[T],
                                    operation: "TreatmentFn[T]",
                                    operation_tag: "TreatmentFn[T]",
                                    preconditions: list[Condition[SimulationPayload[T]]],
                                    postconditions: list[Condition[SimulationPayload[T]]],
                                    **operation_parameters: dict[str, dict]) -> tuple[SimulationPayload[T],
                                                                                      list[CollectedData]]:
    """Managed run conditions and history of a simulator operation. Evaluates the operation."""
    for condition in preconditions:
        if not condition(payload):
            raise ConditionFailed(f'{operation_tag} aborted - condition "{condition}" failed')

    try:
        new_state, new_collected_data = operation(payload.computational_unit)
    except UserWarning as e:
        raise UserWarning(f"Unable to perform operation {operation_tag}, "
                          f"at time point {payload.computational_unit.time}; reason: {e}") from e

    new_state.update_aggregates()

    newpayload: SimulationPayload[T] = SimulationPayload(
        computational_unit=new_state,
        operation_history=payload.operation_history,
        node_id=deepcopy(payload.node_id)
    )

    for condition in postconditions:
        if not condition(newpayload):
            raise ConditionFailed(f'{operation_tag} aborted - condition "{condition}" failed')

    newpayload.operation_history.append((payload.computational_unit.time, operation_tag, operation_parameters))

    return newpayload, new_collected_data
