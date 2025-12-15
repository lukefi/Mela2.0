from typing import Any, Callable, Optional
from copy import deepcopy
from lukefi.metsi.app.utils import ConditionFailed
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.collected_data import CollectedData
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.treatment import FinalTreatment, Treatment


def _build_final_treatment[T: ComputationalUnit](
    *,
    stand: T,
    treatment: Treatment[T],
    event_tags: Optional[set[str]],
    base_params: Optional[dict[str, Any]],
    dynamic_parameters: Optional[dict[str, Callable[[T], Any]]],
) -> tuple[FinalTreatment[T], dict[str, Any]]:
    """
    Resolve and merge treatment parameters and return an executable FinalTreatment.

    - `base_params` should already contain static + file parameters.
    - `dynamic_parameters` maps parameter name -> callable, evaluated against `stand`
      right before execution.
    - Dynamic parameters override `base_params` on key collisions.

    Returns:
        (prepared_treatment, combined_params)
    """
    resolved_dynamic: dict[str, Any] = {}
    if dynamic_parameters:
        resolved_dynamic = {k: fn(stand) for k, fn in dynamic_parameters.items()}

    combined_params = {**(base_params or {}), **resolved_dynamic}

    prepared = FinalTreatment(
        treatment,
        event_tags=event_tags,
        **combined_params,
    )
    return prepared, combined_params


def processor[T: ComputationalUnit](
    payload: SimulationPayload[T],
    treatment: Treatment[T],
    preconditions: list[Condition[T]],
    postconditions: list[Condition[T]],
    *,
    event_tags: Optional[set[str]] = None,
    base_params: Optional[dict[str, Any]] = None,  # static + file already merged
    dynamic_parameters: Optional[dict[str, Callable[[T], Any]]] = None,
) -> tuple[SimulationPayload[T], list[CollectedData]]:
    """Managed run conditions and history of a simulator operation. Evaluates the operation."""

    for condition in preconditions:
        if not condition(payload):
            raise ConditionFailed(f'Treatment {treatment.name} aborted - precondition "{condition}" failed')

    stand = payload.computational_unit

    prepared, combined_params = _build_final_treatment(
        stand=stand,
        treatment=treatment,
        event_tags=event_tags,
        base_params=base_params,
        dynamic_parameters=dynamic_parameters,
    )

    try:
        new_state, new_collected_data = prepared(stand)
    except UserWarning as e:
        raise UserWarning(
            f"Unable to perform treatment {prepared}, at time point {stand.time}; reason: {e}"
        ) from e

    new_state.update_aggregates()

    newpayload: SimulationPayload[T] = SimulationPayload(
        computational_unit=new_state,
        operation_history=payload.operation_history,
        node_id=deepcopy(payload.node_id)
    )

    for condition in postconditions:
        if not condition(newpayload):
            raise ConditionFailed(
                f'Treatment {prepared} aborted - postcondition "{condition}" failed'
            )

    newpayload.operation_history.append(
        (stand.time, prepared.name, combined_params, prepared.tags)
    )

    return newpayload, new_collected_data
