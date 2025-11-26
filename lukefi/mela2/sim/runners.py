from collections.abc import Callable
from typing import TypeVar
from lukefi.metsi.data.computational_unit import ComputationalUnit


T = TypeVar("T", bound=ComputationalUnit)


def evaluate_sequence[V](payload: V, *operations: Callable[[V], V]) -> V:
    """
    Compute a single processing result for single data input.

    Execute all given instruction functions, chaining the input_data argument and
    iterative results as arguments to subsequent calls. Abort on any function raising an exception.

    :param payload: argument for the first instruction functions
    :param operations: *arg list of operation functions to execute in order
    :raises Exception: on any instruction function raising, catch and propagate the exception
    :return: return value of the last instruction function
    """
    current = payload
    for func in operations:
        current = func(current)
    return current
