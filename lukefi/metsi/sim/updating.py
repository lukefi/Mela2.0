
import sqlite3
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.sim_configuration import UpdatingInstructions
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.treatment import PredeterminedTreatment


def update_units[T: ComputationalUnit](updating_instructions: UpdatingInstructions[T],
                                       units: list[T],
                                       db: sqlite3.Connection | None = None) -> list[SimulationPayload[T]]:
    target_time = updating_instructions.target_time
    transition = updating_instructions.transition

    retval = []
    for unit in units:
        if unit.time > target_time:
            raise MetsiException(f"Unable to update unit {unit.identifier} to time {target_time}: "
                                 f"unit already at {unit.time}.")

        current = SimulationPayload(unit)
        while current.computational_unit.time < target_time:
            step, treatments = get_step_and_treatments(current.computational_unit, target_time)

            if step > 0:
                current = SimulationPayload(transition(current.computational_unit, step)[0])

            for treatment in treatments:
                current = SimulationPayload(treatment(current.computational_unit)[0])

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
