
import sqlite3
from typing import Any, Callable, Optional

from lukefi.metsi.sim.collected_data import CollectableDataTypes, OpTuple
from lukefi.metsi.sim.db_utils import NodeType, output_node_to_db
from lukefi.metsi.sim.model import ComputationalUnit
from lukefi.metsi.sim.simulation_payload import SimulationPayload


type TransitionFn[T: ComputationalUnit] = Callable[[T, int], OpTuple[T]]
type TransitionInitFn[T: ComputationalUnit] = Callable[[T, dict[str, Any]], None]


class Transition[T: ComputationalUnit]:
    transition_fn: TransitionFn[T]
    parameters: dict[str, Any]
    init_fn: TransitionInitFn[T] | None
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
        *,
        init_fn: TransitionInitFn[T] | None = None,
        **parameters,
    ):
        self.transition_fn = transition_fn
        self.max_step = max_step
        self.parameters = parameters
        self.init_fn = init_fn
        self.db_output_state = db_output_state
        self.db_output_cd = db_output_cd

        if name is not None:
            self.name = name
        else:
            self.name = transition_fn.__name__

        if collected_data is not None:
            self.collected_data = collected_data
        else:
            self.collected_data = set()

    def initialize(self, unit: T) -> None:
        if self.init_fn is not None:
            self.init_fn(unit, self.parameters)

    def __call__(self,
                 payload: SimulationPayload[T],
                 db: Optional[sqlite3.Connection],
                 time_step: int,
                 transition_count: int = 1) -> OpTuple[T]:
        if self.max_step is not None:
            time_step = min(time_step, self.max_step)
        new_state, collected_data = self.transition_fn(payload.computational_unit, time_step, **self.parameters)

        if db is not None and (self.db_output_state or self.db_output_cd):
            output_node_to_db(db,
                              payload.node_id,
                              self.name,
                              self.parameters,
                              new_state,
                              collected_data,
                              output_state=self.db_output_state,
                              output_collected_data=self.db_output_cd,
                              transition_count=transition_count,
                              node_type=NodeType.TRANSITION)

        return new_state, collected_data

    def __str__(self) -> str:
        return f"{{transition_fn: {self.transition_fn.__name__}, " \
            f"max_step: {self.max_step}, parameters: {self.parameters}}}"

    def __repr__(self) -> str:
        return f"{{transition_fn: {self.transition_fn.__name__}, " \
            f"max_step: {self.max_step}, parameters: {self.parameters}}}"
