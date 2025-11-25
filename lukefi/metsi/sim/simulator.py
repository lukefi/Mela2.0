import sqlite3
from typing import Any, Optional
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.event_tree import output_node_to_db
from lukefi.metsi.sim.collected_data import init_collected_data_tables
from lukefi.metsi.sim.sim_configuration import SimConfiguration
from lukefi.metsi.sim.simulation_payload import SimulationPayload


def simulate_alternatives[T: ComputationalUnit](control: dict[str, Any],
                                                units: list[T],
                                                db: Optional[sqlite3.Connection] = None):
    simconfig = SimConfiguration[T](control["simulation_instructions"], control["transition"], control["end_condition"])

    if db is not None:
        init_collected_data_tables(db, simconfig.collected_data)

    for unit in units:
        payload = SimulationPayload(unit)
        if db is not None:
            output_node_to_db(db, payload, [])
        _simulate_unit(payload, simconfig, db)


def _simulate_unit[T: ComputationalUnit](payload: SimulationPayload[T],
                                         config: SimConfiguration[T],
                                         db: Optional[sqlite3.Connection] = None) -> list[SimulationPayload[T]]:
    retval = []
    if not config.end_condition(payload.computational_unit):
        offset = 0
        for instruction in config.instructions:
            if all(condition(payload) for condition in instruction.conditions):
                for new_branch in instruction.unwrap(payload, offset, db):
                    offset += 1
                    new_branch.computational_unit, _ = config.transition(new_branch.computational_unit)
                    retval.extend(_simulate_unit(new_branch, config, db))
            else:
                # Conditions failed. Don't kill the branch but carry on with transition.
                payload.computational_unit, _ = config.transition(payload.computational_unit)
                retval.extend(_simulate_unit(payload, config, db))
    else:
        retval = [payload]
    return retval
