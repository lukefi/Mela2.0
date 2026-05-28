
import sqlite3
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.domain.utils.file_io import output_node_to_db
from lukefi.metsi.sim.sim_configuration import UpdatingInstructions
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.treatment import PredeterminedTreatment


def update_units[T: ComputationalUnit](updating_instructions: UpdatingInstructions[T],
                                       units: list[T],
                                       db: sqlite3.Connection | None = None) -> list[SimulationPayload[T]]:
    target_time = updating_instructions.target_time
    transition = updating_instructions.transition
    output_transition_state = updating_instructions.output_transition_state
    output_transition_cd = updating_instructions.output_transition_cd
    output_treatment_state = updating_instructions.output_treatment_state
    output_treatment_cd = updating_instructions.output_treatment_cd

    retval = []
    for unit in units:
        if unit.time > target_time:
            raise MetsiException(f"Unable to update unit {unit.identifier} to time {target_time}: "
                                 f"unit already at {unit.time}.")

        current = SimulationPayload(unit)
        while current.computational_unit.time < target_time:
            step, treatments = get_step_and_treatments(current.computational_unit, target_time)

            if step > 0:
                current.computational_unit, cd = transition(current.computational_unit, step)
                if db is not None:
                    output_node_to_db(
                        db,
                        current.node_id,
                        transition.__name__,
                        {},
                        current.computational_unit,
                        cd,
                        tags=None,
                        output_state=output_transition_state,
                        output_collected_data=output_transition_cd,
                        transition_count=1)

            for treatment in treatments:
                current.computational_unit, cd = treatment(current.computational_unit)
                current.node_id.append(0)
                if db is not None:
                    output_node_to_db(
                        db,
                        current.node_id,
                        treatment.name,
                        treatment.evaluated_params,
                        current.computational_unit,
                        cd,
                        treatment.tags,
                        output_state=output_treatment_state,
                        output_collected_data=output_treatment_cd
                    )

        # Update starting time for relative timepoints
        current.computational_unit.start_time = current.computational_unit.time
        retval.append(current)

    return retval


def get_step_and_treatments[T: ComputationalUnit](unit: T,
                                                  target_time: int) -> tuple[int, list[PredeterminedTreatment[T]]]:
    if unit.predetermined_treatments is not None:
        for treatment in unit.predetermined_treatments:
            if unit.time < treatment.time <= target_time:
                return (treatment.time - unit.time,
                        [treatment_ for treatment_ in unit.predetermined_treatments
                         if treatment_.time == treatment.time])
    return target_time - unit.time, []
