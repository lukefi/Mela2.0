import ast
from copy import deepcopy
import csv
import sqlite3
from typing import Any, Set, TypeVar

from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.domain.utils.file_io import NodeType, output_node_to_db
from lukefi.metsi.sim.collected_data import init_collected_data_tables
from lukefi.metsi.sim.instructions import ResimulationInstructions
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.treatment import PredeterminedTreatment, TreatmentFn
from lukefi.metsi.sim.updating import get_step_and_treatments

T = TypeVar("T", bound=ComputationalUnit)
Schedule = list[tuple[int, PredeterminedTreatment[T]]]
Schedules = list[Schedule[T]]


class UnitInstructions[T: ComputationalUnit]:
    initial_state: T
    schedules: Schedules[T]

    def __init__(self,
                 initial_state: T) -> None:
        self.initial_state = initial_state
        self.schedules = []


def resimulate_schedules(resim_instructions: ResimulationInstructions[T],
                         in_db: sqlite3.Connection,
                         out_db: sqlite3.Connection):
    schedules_file_path = resim_instructions.schedules_file

    treatment_map = resim_instructions.treatment_map
    collected_data_types = resim_instructions.collected_data
    init_collected_data_tables(out_db, collected_data_types)
    transition = resim_instructions.transition
    instructions_per_stand = _recreate_instructions(
        schedules_file_path, treatment_map, in_db, resim_instructions.data_type)

    for stand, instructions in instructions_per_stand.items():
        print(f"Resimulating stand {stand}, start year {instructions.initial_state.time}")
        for i, schedule in enumerate(instructions.schedules):
            print(f"Schedule {i}:")
            current = SimulationPayload(deepcopy(instructions.initial_state))
            current.node_id[0] = i
            current.computational_unit.predetermined_treatments = schedule
            target_time = max(time for time, _ in schedule)
            should_run = True
            while should_run:
                step, treatments = get_step_and_treatments(current.computational_unit, target_time)
                print(f"Year {current.computational_unit.time}")
                print(f"treatments: {[treatment.name for treatment in treatments]}")

                for treatment in treatments:
                    print(f"Perform treatment {treatment.name}")
                    current.computational_unit, cd = treatment(current.computational_unit)
                    current.computational_unit.update_aggregates()
                    current.node_id.append(0)
                    output_node_to_db(
                        out_db,
                        current.node_id,
                        treatment.name,
                        treatment.evaluated_params,
                        current.computational_unit,
                        cd,
                        treatment.tags,
                        output_state=True,
                        output_collected_data=True,
                        node_type=NodeType.RESIMULATION_TREATMENT if step > 0 else NodeType.RESIMULATION_TREATMENT_LEAF
                    )

                if step > 0:
                    print(f"Transition to {current.computational_unit.time + step}, step {step}")
                    current.computational_unit, cd = transition(current.computational_unit, step)
                    current.computational_unit.update_aggregates()
                    output_node_to_db(
                        out_db,
                        current.node_id,
                        transition.__name__,
                        {},
                        current.computational_unit,
                        cd,
                        tags=None,
                        output_state=True,
                        output_collected_data=True,
                        transition_count=1,
                        node_type=NodeType.RESIMULATION_TRANSITION
                    )
                else:
                    should_run = False


def _recreate_instructions(sched_file_path: str,
                           treatment_map: dict[str, TreatmentFn[T]],
                           in_db: sqlite3.Connection,
                           data_type: type[T]) -> dict[str, UnitInstructions[T]]:
    instructions_per_stand: dict[str, UnitInstructions[T]] = {}
    with open(sched_file_path, "r", encoding="utf-8") as sched_file:
        sched_reader = csv.DictReader(sched_file, delimiter=";")
        for schedule_row in sched_reader:
            stand_id, treatments = _recreate_schedule(schedule_row, treatment_map, in_db)
            instructions_per_stand.setdefault(
                stand_id,
                UnitInstructions(data_type.reconstruct_initial_state(stand_id, in_db))
            ).schedules.append(treatments)
    return instructions_per_stand


def _recreate_schedule(schedule_row: dict[str, str],
                       treatment_map: dict[str, TreatmentFn],
                       in_db: sqlite3.Connection) -> tuple[str, Schedule]:
    unit_id, leaf_node_id = _parse_schedule_row(schedule_row)
    print(f"Recreating schedule for {unit_id}: {leaf_node_id}")
    cur = in_db.cursor()
    cur.row_factory = sqlite3.Row
    cur.execute(
        """--sql
        WITH RECURSIVE schedule(node, stand, leaf_node) AS (
            SELECT identifier, stand, identifier FROM nodes
            WHERE node_type = 3
            UNION
            SELECT
                nodes.identifier,
                nodes.stand,
                schedule.leaf_node
            FROM nodes, schedule
            WHERE
                schedule.node LIKE nodes.identifier || '%' AND
                nodes.stand = schedule.stand
        )
        SELECT
            stands.year,
            nodes.done_treatment,
            nodes.treatment_params,
            nodes.tags
        FROM schedule, nodes, stands
        WHERE
            nodes.identifier = schedule.node AND
            nodes.stand = schedule.stand AND
            nodes.stand = stands.identifier AND
            nodes.identifier = stands.node AND
            nodes.stand = ? AND
            schedule.leaf_node = ?
        ORDER BY leaf_node, schedule.node;
        """,
        (
            unit_id,
            leaf_node_id
        )
    )

    retval: Schedule = []
    for step in cur:
        retval.append(
            (
                step["year"],
                PredeterminedTreatment(
                    name=step["done_treatment"],
                    treatment_fn=treatment_map[step["done_treatment"]],
                    tags=_parse_tags(step["tags"]),
                    # Evaluate all parameters as static for now
                    static_parameters=_parse_params(step["treatment_params"])
                )
            )
        )
        print(f"{step['year']}: {step['done_treatment']} ({step['tags']})")
    return unit_id, retval


def _parse_schedule_row(schedule_row: dict[str, str]) -> tuple[str, str]:
    return schedule_row["standId"], schedule_row["schedId"]


def _parse_tags(tags_str: str) -> Set[str]:
    return ast.literal_eval(tags_str)


def _parse_params(params_str: str) -> dict[str, Any]:
    return ast.literal_eval(params_str)
