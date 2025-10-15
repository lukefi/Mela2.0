from collections.abc import Callable
import sqlite3
from typing import Optional, TypeVar
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.collected_data import CollectedData
from lukefi.metsi.sim.event_tree import EventTree
from lukefi.metsi.sim.generators import Generator

from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.sim_configuration import SimConfiguration

T = TypeVar("T", bound=ComputationalUnit)

Runner = Callable[[list[T], SimConfiguration[T], Optional[sqlite3.Connection]], None]


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


def run_full_tree_strategy(payload: SimulationPayload[T],
                           config: SimConfiguration,
                           db: Optional[sqlite3.Connection] = None) -> None:
    """Process the given operation payload using a simulation state tree created from the declaration. Full simulation
    tree and operation chains are pre-generated for the run. This tree strategy creates the full theoretical branching
    tree for the simulation, carrying a significant memory and runtime overhead for large trees.

    :param payload: a simulation state payload
    :param config: a prepared SimConfiguration object
    :param evaluator: a function for performing computation from given EventTree and for given OperationPayload
    :return: a list of resulting simulation state payloads
    """

    nestable_generator: Generator[T] = config.full_tree_generators()
    collected_data = nestable_generator.get_types_of_collected_data()
    root_node: EventTree[T] = nestable_generator.compose_nested()
    return root_node.evaluate(payload, [0], db)
