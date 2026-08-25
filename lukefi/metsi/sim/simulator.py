from copy import copy
import sqlite3
from typing import Any, Optional, cast
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.domain.utils.file_io import NodeType, output_node_to_db, update_leaf_node
from lukefi.metsi.sim.collected_data import CollectableDataTypes, init_collected_data_tables
from lukefi.metsi.sim.sim_configuration import SimConfiguration
from lukefi.metsi.sim.instructions import SimulationInstruction
from lukefi.metsi.sim.simulation_payload import SimulationPayload


def simulate_alternatives[T: ComputationalUnit](control: dict[str, Any],
                                                units: list[T] | list[SimulationPayload[T]],
                                                db: Optional[sqlite3.Connection] = None,
                                                existing_data_types: CollectableDataTypes | None = None):
    simconfig = SimConfiguration[T](
         control["simulation_instructions"],
         control["transition"],
         control["end_condition"],
         control.get("initialization", None))

    if db is not None:
        init_collected_data_tables(db, simconfig.collected_data, existing_data_types)

    for i, unit in enumerate(units, 1):
        if not isinstance(unit, SimulationPayload):
            # ensure type of SimulationPayload[T]
            payload = cast(SimulationPayload[T], SimulationPayload(unit))
        else:
            payload = unit
        if simconfig.initialization is not None:
            payload.computational_unit.update_aggregates()
            # initialize transition state
            simconfig.initialization(payload.computational_unit)
        print(f"Simulating unit {payload.computational_unit.identifier} ({i} of {len(units)})...")
        payload.computational_unit.update_aggregates()

        if db is not None:
            # Write initial state to database
            output_node_to_db(db,
                              payload.node_id,
                              "do_nothing",
                              {},
                              payload.computational_unit,
                              [],
                              {"initial"},
                              node_type=NodeType.INITIAL)
        _simulate_unit(payload, simconfig, db)


def _simulate_unit[T: ComputationalUnit](payload: SimulationPayload[T],
                                         config: SimConfiguration[T],
                                         db: Optional[sqlite3.Connection] = None,
                                         transition_count: int = 0):
    if not config.end_condition(payload):
        offset = 0
        all_instructions_failed = True
        for instruction in config.instructions:
            if all(condition(payload) for condition in instruction.conditions):
                all_instructions_failed = False
                for new_branch in instruction.evaluate(copy(payload), db, node=offset):
                    time_step = _find_next_time_step(
                        new_branch,
                        config.instructions,
                        config.transition.max_step)
                    new_branch.computational_unit, _ = config.transition(new_branch, db, time_step)
                    new_branch.computational_unit.update_aggregates()
                    _simulate_unit(new_branch, config, db, 1)
                offset += instruction.event_generator.width()
        if all_instructions_failed:
            # All instructions had failed conditions. Create one branch to carry on with transition.
            time_step = _find_next_time_step(
                payload,
                config.instructions,
                config.transition.max_step)
            transition_count += 1
            payload.computational_unit, _ = config.transition(payload, db, time_step, transition_count)
            payload.computational_unit.update_aggregates()
            _simulate_unit(payload, config, db, transition_count)
    else:
        # End condition met, update `leaf` column
        if db is not None:
            update_leaf_node(db, payload, transition_count)


def _find_next_time_step[T: ComputationalUnit](payload: SimulationPayload[T],
                                               instructions: list[SimulationInstruction[T]],
                                               maximum_step: int):
    current_time = payload.computational_unit.time
    time_points: set[int] = set()
    for instruction in instructions:
        time_points.update(
            filter(
                lambda t: t > current_time,
                instruction.time_points(
                    payload.computational_unit.start_time)))

    if time_points:
        next_time_point = min(time_points)
        return next_time_point - current_time
    return maximum_step
