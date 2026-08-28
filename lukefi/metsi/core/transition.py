
import sqlite3
from typing import Any, Callable, Optional

from lukefi.metsi.core.collected_data import CollectableDataTypes, OpTuple
from lukefi.metsi.core.db_utils import NodeType, output_node_to_db
from lukefi.metsi.core.model import ComputationalUnit
from lukefi.metsi.core.simulation_payload import SimulationPayload


type TransitionFn[T: ComputationalUnit] = Callable[[T, int], OpTuple[T]]
type TransitionInitFn[T: ComputationalUnit] = Callable[[T], None]

class Initialization[T: ComputationalUnit]:
    init_fn: TransitionInitFn[T]
    params: dict[str, Any]

    def __init__(self, init_fn: TransitionInitFn[T], init_params: dict[str, Any] | None = None) -> None:
        self.init_fn = init_fn
        self.params = init_params or {}

    def __call__(self, unit: T):
        self.init_fn(unit, **self.params)


class Transition[T: ComputationalUnit]:

    __slots__ = ("transition_fn",
                 "parameters",
                 "initialization",
                 "name",
                 "collected_data",
                 "db_output_state",
                 "db_output_cd",
                 "max_step")

    transition_fn: TransitionFn[T]
    parameters: dict[str, Any]
    initialization: Initialization[T] | None
    name: str
    collected_data: CollectableDataTypes
    db_output_state: bool
    db_output_cd: bool
    max_step: int

    def __init__(
        self,
        transition_fn: TransitionFn[T],
        max_step: int = 5,
        collected_data: Optional[CollectableDataTypes] = None,
        name: Optional[str] = None,
        db_output_state: bool = False,
        db_output_cd: bool = True,
        initialization: Optional[Initialization[T]] = None,
        **parameters,
    ):
        self.transition_fn = transition_fn
        self.max_step = max_step
        self.parameters = parameters
        self.db_output_state = db_output_state
        self.db_output_cd = db_output_cd
        self.initialization = initialization
        self.parameters = parameters

        if name is not None:
            self.name = name
        else:
            self.name = transition_fn.__name__

        if collected_data is not None:
            self.collected_data = collected_data
        else:
            self.collected_data = set()

    def __call__(self,
                 payload: SimulationPayload[T],
                 db: Optional[sqlite3.Connection],
                 time_step: int,
                 transition_count: int = 1) -> OpTuple[T]:
        if self.max_step is not None:
            time_step = min(time_step, self.max_step)
        new_state, collected_data = self.transition_fn(payload.unit, time_step, **self.parameters)

        if db is not None and (self.db_output_state or self.db_output_cd):
            output_node_to_db(db,
                              payload.node_id,
                              self.name,
                              self.parameters,
                              new_state,
                              collected_data,
                              output_state=self.db_output_state,
                              output_cd=self.db_output_cd,
                              transition_count=transition_count,
                              node_type=NodeType.TRANSITION)

        return new_state, collected_data

    def __str__(self) -> str:
        return f"{{transition_fn: {self.transition_fn.__name__}, " \
            f"max_step: {self.max_step}, parameters: {self.parameters}}}"

    def __repr__(self) -> str:
        return f"{{transition_fn: {self.transition_fn.__name__}, " \
            f"max_step: {self.max_step}, parameters: {self.parameters}}}"
