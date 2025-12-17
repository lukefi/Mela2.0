from copy import deepcopy
from lukefi.metsi.app.utils import ConditionFailed
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.collected_data import CollectedData
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.treatment import PreparedTreatment


def processor[T: ComputationalUnit](payload: SimulationPayload[T],
                                    treatment: PreparedTreatment[T],
                                    preconditions: list[Condition[T]],
                                    postconditions: list[Condition[T]],
                                    **operation_parameters: dict[str, dict]) -> tuple[SimulationPayload[T],
                                                                                      list[CollectedData]]:
    """Managed run conditions and history of a simulator operation. Evaluates the operation."""
    for condition in preconditions:
        if not condition(payload):
            raise ConditionFailed(f'Treatment {treatment} aborted - precondition "{condition}" failed')

    try:
        new_state, new_collected_data = treatment(payload.computational_unit)
    except UserWarning as e:
        raise UserWarning(f"Unable to perform treatment {treatment}, "
                          f"at time point {payload.computational_unit.time}; reason: {e}") from e

    new_state.update_aggregates()

    newpayload: SimulationPayload[T] = SimulationPayload(
        computational_unit=new_state,
        operation_history=payload.operation_history,
        node_id=deepcopy(payload.node_id)
    )

    for condition in postconditions:
        if not condition(newpayload):
            raise ConditionFailed(f'Treatment {treatment} aborted - postcondition "{condition}" failed')

    newpayload.operation_history.append(
        (payload.computational_unit.time,
         treatment.name,
         operation_parameters,
         treatment.tags))

    return newpayload, new_collected_data
