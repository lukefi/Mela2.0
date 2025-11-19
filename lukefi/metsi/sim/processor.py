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

def _resolve_dynamic_tree(obj: Any, stand: T) -> Any:
    """
    Recursively resolve a tree of dynamic parameters.

    - If value is callable: call it with `stand` and use the result.
    - If dict/list/tuple: recurse into children.
    - Otherwise: return value as-is.
    """
    if callable(obj):
        return obj(stand)
    if isinstance(obj, dict):
        return {k: _resolve_dynamic_tree(v, stand) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_dynamic_tree(v, stand) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_resolve_dynamic_tree(v, stand) for v in obj)
    return obj

def _merge_params(base: Any, overlay: Any) -> Any:
    """
    Deep-merge two parameter trees.

    - For dicts: merge keys, overlay wins, recurse on common keys.
    - For lists: merge by index, overlay items overwrite base items.
    - For everything else: overlay replaces base.
    """
    if isinstance(base, dict) and isinstance(overlay, dict):
        result = dict(base)
        for k, v in overlay.items():
            if k in result:
                result[k] = _merge_params(result[k], v)
            else:
                result[k] = v
        return result

    if isinstance(base, list) and isinstance(overlay, list):
        result = list(base)
        for i, v in enumerate(overlay):
            if i < len(result):
                result[i] = _merge_params(result[i], v)
            else:
                result.append(v)
        return result

    # scalar / mismatched types -> overlay replaces base
    return overlay

def processor(payload: SimulationPayload[T],
                                    operation: "TreatmentFn[T]",
                                    operation_tag: "TreatmentFn[T]",
                                    time_point: int,
                                    preconditions: list[Condition[SimulationPayload[T]]],
                                    postconditions: list[Condition[SimulationPayload[T]]],
                                    *,
                                    static_params: dict[str, Any] | None = None,
                                    dynamic_params: dict[str, Any] | None = None,
                                ) -> tuple[SimulationPayload[T], list[CollectedData]]:
    """Managed run conditions and history of a simulator operation. Evaluates the operation."""
    for condition in preconditions:
        if not condition(time_point, payload):
            raise ConditionFailed(f'{operation_tag} aborted - condition "{condition}" failed')

    payload.computational_unit.update_aggregates()
    final_params: dict[str, Any] = dict(static_params or {})

    if dynamic_params:
        resolved_dynamic = _resolve_dynamic_tree(dynamic_params, payload.computational_unit)
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
        operation_history=payload.operation_history
    )

    for condition in postconditions:
        if not condition(time_point, newpayload):
            raise ConditionFailed(f'{operation_tag} aborted - condition "{condition}" failed')

    # store the params that were actually used (static + resolved dynamic)
    newpayload.operation_history.append((time_point, operation_tag, final_params))

    return newpayload, new_collected_data
