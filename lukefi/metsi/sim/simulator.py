import sqlite3
from typing import Any, Optional
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.event_tree import output_node_to_db
from lukefi.metsi.sim.collected_data import init_collected_data_tables
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.sim_configuration import SimConfiguration, TransitionFn
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.simulation_payload import SimulationPayload


def simulate_alternatives[T: ComputationalUnit](control: dict[str, Any],
                                                units: list[T],
                                                db: Optional[sqlite3.Connection] = None):
    simconfig = SimConfiguration[T](control["simulation_instructions"], control["transition"], control["end_condition"])

    if db is not None:
        init_collected_data_tables(db, simconfig.collected_data)

    instructions = simconfig.instructions
    end_condition = simconfig.end_condition
    transition = simconfig.transition

    for unit in units:
        payload = SimulationPayload(unit)
        if db is not None:
            payload.node_id.append(0)
            output_node_to_db(db, payload, [])
        _simulate_unit(payload, transition.transition_fn, end_condition, instructions, db)


def _simulate_unit[T: ComputationalUnit](payload: SimulationPayload[T],
                                         transition: TransitionFn[T],
                                         end_condition: Condition[T],
                                         instructions: list[SimulationInstruction[T]],
                                         db: Optional[sqlite3.Connection] = None) -> list[SimulationPayload[T]]:
    retval = []
    if not end_condition(payload.computational_unit):
        offset = 0
        for instruction in instructions:
            if all(condition(payload) for condition in instruction.conditions):
                for new_branch in instruction.unwrap(payload, offset, db):
                    offset += 1
                    new_branch.computational_unit, _ = transition(new_branch.computational_unit)
                    retval.extend(_simulate_unit(new_branch, transition, end_condition, instructions, db))
            else:
                # Conditions failed. Don't kill the branch but carry on with transition.
                payload.computational_unit, _ = transition(payload.computational_unit)
                retval.extend(_simulate_unit(payload, transition, end_condition, instructions, db))
    else:
        retval = [payload]
    return retval
