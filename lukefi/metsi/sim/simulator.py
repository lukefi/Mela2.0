from copy import copy
import sqlite3
from typing import Optional, Sequence, cast
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.domain.utils.file_io import NodeType, output_node_to_db, update_leaf_node
from lukefi.metsi.sim.collected_data import CollectableDataTypes, init_collected_data_tables
from lukefi.metsi.sim.instructions import SimulationInstruction
from lukefi.metsi.sim.sim_control import Simulation
from lukefi.metsi.sim.simulation_payload import SimulationPayload


def simulate_alternatives[T: ComputationalUnit](control: Simulation[T],
                                                units: Sequence[T] | Sequence[SimulationPayload[T]],
                                                db: Optional[sqlite3.Connection] = None,
                                                existing_data_types: CollectableDataTypes | None = None):
    if db is not None:
        init_collected_data_tables(db, control.collected_data, existing_data_types)

    for i, unit in enumerate(units, 1):
        if not isinstance(unit, SimulationPayload):
            payload = cast(SimulationPayload[T], SimulationPayload(unit))
        else:
            payload = unit
        print(f"Simulating unit {payload.computational_unit.identifier} ({i} of {len(units)})...")
        payload.computational_unit.update_aggregates()
        control.transition.initialize(payload.computational_unit)

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
        _simulate_unit(payload, control, db)


def _simulate_unit[T: ComputationalUnit](payload: SimulationPayload[T],
                                         control: Simulation[T],
                                         db: Optional[sqlite3.Connection] = None,
                                         transition_count: int = 0):
    if not control.end_condition(payload):
        offset = 0
        all_instructions_failed = True
        for instruction in control.instructions:
            if all(condition(payload) for condition in instruction.conditions):
                all_instructions_failed = False
                for new_branch in instruction.evaluate(copy(payload), db, node=offset):
                    time_step = _find_next_time_step(
                        new_branch,
                        control.instructions,
                        control.transition.max_step)
                    new_branch.computational_unit, _ = control.transition(new_branch, db, time_step)
                    new_branch.computational_unit.update_aggregates()
                    _simulate_unit(new_branch, control, db, 1)
                offset += instruction.event_generator.width()
        if all_instructions_failed:
            # All instructions had failed conditions. Create one branch to carry on with transition.
            time_step = _find_next_time_step(
                payload,
                control.instructions,
                control.transition.max_step)
            transition_count += 1
            payload.computational_unit, _ = control.transition(payload, db, time_step, transition_count)
            payload.computational_unit.update_aggregates()
            _simulate_unit(payload, control, db, transition_count)
    else:
        # End condition met, update `leaf` column
        if db is not None:
            update_leaf_node(db, payload, transition_count)


def _find_next_time_step[T: ComputationalUnit](payload: SimulationPayload[T],
                                               instructions: Sequence[SimulationInstruction[T]],
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
