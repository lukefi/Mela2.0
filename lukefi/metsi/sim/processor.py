from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar
from lukefi.metsi.app.utils import ConditionFailed
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.collected_data import CollectedData
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.operations import prepared_treatment
if TYPE_CHECKING:
    from lukefi.metsi.sim.generators import TreatmentFn

T = TypeVar("T", bound=ComputationalUnit)


def _merge_params(base: Any, overlay: Any) -> Any:
    """
    Deep-merge two parameter trees.

    - For dicts: merge keys, overlay wins, recurse on common keys.
    - For lists: merge by index, overlay items overwrite base items.
    - For everything else: overlay replaces base.
    """
    if isinstance(base, dict) and isinstance(overlay, dict):
        result_dict = dict(base)
        for k, v in overlay.items():
            if k in result_dict:
                result_dict[k] = _merge_params(result_dict[k], v)
            else:
                result_dict[k] = v
        return result_dict

    if isinstance(base, list) and isinstance(overlay, list):
        result_list = list(base)
        for i, v in enumerate(overlay):
            if i < len(result_list):
                result_list[i] = _merge_params(result_list[i], v)
            else:
                result_list.append(v)
        return result_list

    # scalar / mismatched types -> overlay replaces base
    return overlay


def processor(
    payload: SimulationPayload[T],
    operation: "TreatmentFn[T]",
    operation_tag: "TreatmentFn[T]",
    time_point: int,
    preconditions: list[Condition[SimulationPayload[T]]],
    postconditions: list[Condition[SimulationPayload[T]]],
    *,
    static_params: dict[str, Any] | None = None,
    dynamic_params: dict[str, Callable[[T], Any]] | None = None,
) -> tuple[SimulationPayload[T], list[CollectedData]]:
    """
    Managed run conditions and history of a simulator operation.

    - static_params: plain, stand-independent parameters.
    - dynamic_params: mapping name -> fn(stand) returning the full
      value for that parameter (can be nested dict/list/…).
    """
    # --- Preconditions ---
    for condition in preconditions:
        if not condition(time_point, payload):
            raise ConditionFailed(f'{operation_tag} aborted - condition "{condition}" failed')

    # Make sure aggregates are up-to-date before computing dynamic params
    payload.computational_unit.update_aggregates()
    final_params: dict[str, Any] = dict(static_params or {})

    # --- Resolve dynamic parameters (flat dict[str, Callable[[T], Any]]) ---
    if dynamic_params:
        stand = payload.computational_unit
        resolved_dynamic: dict[str, Any] = {}
        for name, param_fn in dynamic_params.items():
            if not callable(param_fn):
                # dynamic params should be callables
                raise TypeError(
                    f"Dynamic parameter {name!r} is not callable; "
                    "expected Callable[[T], Any]."
                )
            resolved_dynamic[name] = param_fn(stand)

        # Deep-merge static + resolved dynamic
        final_params = _merge_params(final_params, resolved_dynamic)

    bound_operation = prepared_treatment(operation, **final_params)

    try:
        new_state, new_collected_data = bound_operation(payload.computational_unit)
    except UserWarning as e:
        raise UserWarning(
            f"Unable to perform operation {operation_tag}, "
            f"at time point {time_point}; reason: {e}"
        ) from e

    new_state.update_aggregates()

    newpayload: SimulationPayload[T] = SimulationPayload(
        computational_unit=new_state,
        operation_history=payload.operation_history,
    )

    # --- Postconditions ---
    for condition in postconditions:
        if not condition(time_point, newpayload):
            raise ConditionFailed(f'{operation_tag} aborted - condition "{condition}" failed')

    # store the params that were actually used (static + resolved dynamic)
    newpayload.operation_history.append((time_point, operation_tag, final_params))

    return newpayload, new_collected_data
