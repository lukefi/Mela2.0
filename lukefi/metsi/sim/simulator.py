import sqlite3
from typing import Any, Optional
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.collected_data import init_collected_data_tables
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.sim_configuration import SimConfiguration, TransitionFn
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.simulation_payload import SimulationPayload


def simulate_alternatives[T: ComputationalUnit](control: dict[str, Any],
                                                units: list[T],
                                                db: Optional[sqlite3.Connection] = None):
    simconfig = SimConfiguration[T](**control)

    if db is not None:
        init_collected_data_tables(db, simconfig.collected_data)

    instructions = simconfig.instructions
    end_condition = simconfig.end_condition
    transition = simconfig.transition

    for unit in units:
        payload = SimulationPayload(unit)
        _simulate(payload, transition.transition_fn, end_condition, instructions)


def _simulate[T: ComputationalUnit](branch: SimulationPayload[T],
                                    transition: TransitionFn[T],
                                    end_condition: Condition[T],
                                    instructions: list[SimulationInstruction[T]]):
    if not end_condition(branch.computational_unit):
        # branch.computational_unit = transition(branch.computational_unit)
        for instruction in instructions:
            for condition in instruction.conditions:
                if not condition(branch):
                    continue
            for new_branch in instruction.unwrap(branch):
                new_branch.computational_unit = transition(new_branch.computational_unit)
                _simulate(new_branch, transition, end_condition, instructions)
