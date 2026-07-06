import sqlite3
from typing import Sequence

from lukefi.metsi.sim.collected_data import CollectableDataTypes, init_collected_data_tables
from lukefi.metsi.sim.db_utils import NodeType, output_node_to_db
from lukefi.metsi.sim.exceptions import MetsiException
from lukefi.metsi.sim.model import ComputationalUnit
from lukefi.metsi.sim.sim_control import Updating
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.treatment import PredeterminedTreatment


def update_units[T: ComputationalUnit](updating_instructions: Updating[T],
                                       units: Sequence[T],
                                       db: sqlite3.Connection | None = None) -> tuple[list[SimulationPayload[T]],
                                                                                      CollectableDataTypes | None]:
    target_time = updating_instructions.target_time
    transition = updating_instructions.transition
    output_treatment_state = updating_instructions.output_treatment_state
    output_treatment_cd = updating_instructions.output_treatment_cd
    cd_types: CollectableDataTypes | None = None

    if db is not None:
        cd_types = _get_collected_data_types(units, updating_instructions)
        init_collected_data_tables(db, cd_types)

    retval: list[SimulationPayload[T]] = []
    for unit in units:
        if unit.time > target_time:
            raise MetsiException(f"Unable to update unit {unit.identifier} to time {target_time}: "
                                 f"unit already at {unit.time}.")

        current = SimulationPayload(unit)
        keep_running = True
        while keep_running:
            step, treatments = get_step_and_treatments(current.unit, target_time)

            for treatment in treatments:
                current.unit, cd = treatment(current.unit)
                current.unit.update_aggregates()
                current.node_id.append(0)
                if db is not None:
                    output_node_to_db(
                        db,
                        current.node_id,
                        treatment.name,
                        treatment.evaluated_params,
                        current.unit,
                        cd,
                        treatment.tags,
                        output_state=output_treatment_state,
                        output_cd=output_treatment_cd,
                        node_type=NodeType.UPDATING_TREATMENT
                    )

            if step > 0:
                # TODO: This is a bit of a hack. DB stuff should be refactored.
                current.unit, cd = transition(current, None, step)
                current.unit.update_aggregates()
                if db is not None:
                    output_node_to_db(
                        db,
                        current.node_id,
                        transition.name,
                        {},
                        current.unit,
                        cd,
                        tags=None,
                        output_state=transition.db_output_state,
                        output_cd=transition.db_output_cd,
                        transition_count=1,
                        node_type=NodeType.UPDATING_TRANSITION
                    )
            else:
                keep_running = False

        # Drop performed treatments
        current.unit.predetermined_treatments = None

        # Update starting time for relative timepoints
        current.unit.start_time = current.unit.time

        # TODO: update initial state tables in db?

        retval.append(current)

    return retval, cd_types


def _get_collected_data_types[T: ComputationalUnit](
        units: Sequence[T],
        updating_instructions: Updating) -> CollectableDataTypes:

    if updating_instructions.transition.db_output_cd:
        retval = updating_instructions.transition.collected_data
    else:
        retval = set()

    if updating_instructions.output_treatment_cd:
        for unit in units:
            if unit.predetermined_treatments is not None:
                for _, treatment in unit.predetermined_treatments:
                    retval |= treatment.collected_data

    return retval


def get_step_and_treatments[T: ComputationalUnit](unit: T,
                                                  target_time: int) -> tuple[int, list[PredeterminedTreatment[T]]]:
    if unit.predetermined_treatments is not None:
        treatments = [treatment for time, treatment in unit.predetermined_treatments if time == unit.time]
        for time, _ in unit.predetermined_treatments:
            if unit.time < time <= target_time:
                return time - unit.time, treatments
        return target_time - unit.time, treatments
    return target_time - unit.time, []
