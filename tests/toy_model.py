from sqlite3 import Connection
from typing import override
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.collected_data import CollectedData, OpTuple
from lukefi.metsi.sim.sim_configuration import Transition


class ToyModel(ComputationalUnit):
    value: int

    def __init__(self, identifier: str, value: int, time: int = 0) -> None:
        self.identifier = identifier
        self.value = value
        self.time = time

    @override
    def output_to_db(self, db: Connection, node: str):
        pass

    @override
    def update_aggregates(self):
        pass


class ToyTransition(Transition[ToyModel]):
    def __init__(self, **parameters):
        super().__init__(toy_transition, **parameters)


def toy_transition(state: ToyModel) -> OpTuple[ToyModel]:
    state.time += 1
    return state, []


def toy_inc(x: ToyModel, **operation_params) -> tuple[ToyModel, list[CollectedData]]:
    incrementation = operation_params.get("incrementation", 1)
    x.value += incrementation
    return x, []


def parametrized_treatment(x: ToyModel, **kwargs) -> tuple[ToyModel, list[CollectedData]]:
    if kwargs.get('amplify') is True:
        x.value *= 1000
    return x, []
