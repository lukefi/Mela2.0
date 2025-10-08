import sqlite3
from typing import Optional
from lukefi.metsi.app.console_logging import print_logline
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.forestry_types import ForestOpPayload, StandList
from lukefi.metsi.sim.collected_data import CollectedData
from lukefi.metsi.sim.runners import run_full_tree_strategy
from lukefi.metsi.sim.sim_configuration import SimConfiguration


def run_stands(stands: StandList,
               config: SimConfiguration[ForestStand],
               db: Optional[sqlite3.Connection] = None) -> dict[str, list[ForestOpPayload]]:
    """Run the simulation for all given stands, from the given declaration, using the given runner. Return the
    results organized into a dict keyed with stand identifiers."""

    retval: dict[str, list[ForestOpPayload]] = {}
    for stand in stands:
        overlaid_stand = stand

        payload = ForestOpPayload(
            computational_unit=overlaid_stand,
            collected_data=CollectedData(initial_time_point=config.time_points[0]),
            operation_history=[],
        )

        schedule_payloads = run_full_tree_strategy(payload, config, db)
        identifier = stand.identifier
        print_logline(f"Alternatives for stand {identifier}: {len(schedule_payloads)}")
        retval[identifier] = schedule_payloads
    return retval
