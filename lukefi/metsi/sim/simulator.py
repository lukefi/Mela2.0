import sqlite3
from typing import Any, Optional, TypeVar
from lukefi.metsi.app.console_logging import print_logline
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.domain.collected_data import init_collected_data_tables
from lukefi.metsi.sim.sim_configuration import SimConfiguration
from lukefi.metsi.sim.simulation_payload import SimulationPayload


T = TypeVar("T", bound=ComputationalUnit)


def simulate_alternatives(control: dict[str, Any], units: list[T],
                          db: Optional[sqlite3.Connection] = None) -> dict[str, list[SimulationPayload[T]]]:
    simconfig = SimConfiguration[T](**control)
    nestable_generator = simconfig.full_tree_generators()
    if db is not None:
        collected_data_types = nestable_generator.get_types_of_collected_data()
        init_collected_data_tables(db, collected_data_types)
    root_node = nestable_generator.compose_nested()

    retval: dict[str, list[SimulationPayload[T]]] = {}
    for unit in units:
        payload = SimulationPayload[T](computational_unit=unit, operation_history=[])
        schedule_payloads = root_node.evaluate(payload, [0], db)
        identifier = unit.identifier
        print_logline(f"Alternatives for stand {identifier}: {len(schedule_payloads)}")
        retval[identifier] = schedule_payloads
    return retval
