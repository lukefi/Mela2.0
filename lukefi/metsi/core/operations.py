from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

from lukefi.metsi.core.exceptions import MetsiException
from lukefi.metsi.core.model import ComputationalUnit

if TYPE_CHECKING:
    from lukefi.metsi.core.collected_data import CollectedData


def do_nothing[T: ComputationalUnit](data: T, step: Optional[int] = None, **kwargs) -> tuple[T, list["CollectedData"]]:
    _ = kwargs
    _ = step
    return data, []


def _prepared_operation[T](operation_entrypoint: Callable[[T], T], **operation_parameters) -> Callable[[T], T]:
    """prepares an opertion entrypoint function with configuration parameters"""
    return lambda state: operation_entrypoint(state, **operation_parameters)


def simple_processable_chain[T](operation_tags: Sequence[Callable[[T], T]],
                                operation_params: dict[Callable[[T], T], Any]) -> list[Callable[[T], T]]:
    """Prepare a list of partially applied (parametrized) operation functions based on given declaration of operation
    tags and operation parameters"""
    result: list[Callable[[T], T]] = []
    for tag in operation_tags if operation_tags is not None else []:
        params = operation_params.get(tag, [{}])
        if len(params) > 1:
            raise MetsiException(f"Trying to apply multiple parameter set for preprocessing operation \'{tag}\'. "
                                 "Defining multiple parameter sets is only supported for alternative clause "
                                 "generators.")
        result.append(_prepared_operation(tag, **params[0]))
    return result
