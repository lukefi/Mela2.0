from sqlite3 import Connection
from typing import override
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.collected_data import CollectedData
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.sim_configuration import Transition, TransitionFn
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.simulation_payload import SimulationPayload


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


def toy_transition(state: ToyModel) -> ToyModel:
    state.time += 1
    return state


def toy_inc(x: ToyModel, **operation_params) -> tuple[ToyModel, list[CollectedData]]:
    incrementation = operation_params.get("incrementation", 1)
    x.value += incrementation
    return x, []


def simulate(unit: SimulationPayload[ToyModel],
             transition: TransitionFn[ToyModel],
             end_condition: Condition[ToyModel],
             instructions: list[SimulationInstruction[ToyModel]]) -> list[SimulationPayload[ToyModel]]:
    retval = []
    if not end_condition(unit.computational_unit):
        for instruction in instructions:
            conditions_true = True
            for condition in instruction.conditions:
                if not condition(unit):
                    conditions_true = False
                    break
            if conditions_true:
                for new_branch in instruction.unwrap(unit):
                    new_branch.computational_unit = transition(new_branch.computational_unit)
                    retval.extend(simulate(new_branch, transition, end_condition, instructions))
            else:
                unit.computational_unit = transition(unit.computational_unit)
                retval.extend(simulate(unit, transition, end_condition, instructions))
    else:
        retval = [unit]
    return retval
