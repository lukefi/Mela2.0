from copy import deepcopy
import csv
import sqlite3
from typing import Any, Set

from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.transition import TransitionFn
from lukefi.metsi.sim.treatment import PredeterminedTreatment, TreatmentFn
from lukefi.metsi.sim.updating import get_step_and_treatments

type Schedule = list[tuple[int, PredeterminedTreatment[ForestStand]]]
type Schedules = list[Schedule]


class ResimulationInstructions:
    initial_state: ForestStand
    schedules: Schedules

    def __init__(self,
                 initial_state: ForestStand,
                 schedules: Schedules) -> None:
        self.initial_state = initial_state
        self.schedules = schedules


def resimulate_schedules(control: dict[str, Any],
                         in_db: sqlite3.Connection,
                         out_db: sqlite3.Connection):
    _ = out_db

    # TODO: exact format of schedules csv not fully specified
    schedules_file_path = control["selected_schedules_file"]

    # TODO: recreating dynamic parameters and other complex structures from original control file
    #           - possible quick hack - declare LUT in resim control?
    #           - no need initially - dynamic parameters needed only in Monte-Carlo
    treatment_map = control["treatment_map"]

    transition = _determine_transition(control, in_db)
    instructions_per_stand = _recreate_instructions(schedules_file_path, treatment_map, in_db)

    for instructions in instructions_per_stand.values():
        for schedule in instructions.schedules:
            current = SimulationPayload(deepcopy(instructions.initial_state))
            current.computational_unit.predetermined_treatments = schedule
            time_points = set(time_point for time_point, _ in schedule)
            for time_point in time_points:
                step, treatments = get_step_and_treatments(current.computational_unit, time_point)

                if step > 0:
                    current.computational_unit, cd = transition(current.computational_unit, step)

                for treatment in treatments:
                    current.computational_unit, cd = treatment(current.computational_unit)


def _determine_transition(control: dict[str, Any], in_db: sqlite3.Connection) -> TransitionFn[ForestStand]:
    _ = control
    _ = in_db
    assert False, "_determine_transition not implemented"


def _recreate_instructions(sched_file_path: str,
                           treatment_map: dict[str, TreatmentFn],
                           in_db: sqlite3.Connection) -> dict[str, ResimulationInstructions]:
    # TODO: building simulation instructions (or equivalent) from declared leaf nodes/schedules
    instructions_per_stand: dict[str, ResimulationInstructions] = {}
    with open(sched_file_path, "r", encoding="utf-8") as sched_file:
        sched_reader = csv.reader(sched_file)
        for schedule_row in sched_reader:
            stand_id, treatments = _recreate_schedule(schedule_row, treatment_map, in_db)
            instructions_per_stand.setdefault(
                stand_id,
                ResimulationInstructions(_reconstruct_initial_state(stand_id, in_db), [])
            ).schedules.append(treatments)
    return instructions_per_stand


def _recreate_schedule(schedule_row: list[str],
                       treatment_map: dict[str, TreatmentFn],
                       in_db: sqlite3.Connection) -> tuple[str, Schedule]:
    stand_id, leaf_node_id = _parse_schedule_row(schedule_row)
    cur = in_db.cursor()
    cur.execute(
        """--sql
        WITH RECURSIVE schedule(node, stand, leaf_node) AS (
            SELECT identifier, stand, identifier FROM nodes
            WHERE leaf = 1
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
            --nodes.identifier,
            --nodes.stand,
            stands.year,
            nodes.done_treatment,
            nodes.treatment_params,
            nodes.tags
            --nodes.leaf,
            --schedule.leaf_node
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
            stand_id,
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

    return stand_id, retval


def _parse_schedule_row(schedule_row: list[str]) -> tuple[str, str]:
    return schedule_row[0], schedule_row[1]


def _parse_tags(tags_str: str) -> Set[str]:
    _ = tags_str
    assert False, "_parse_tags not implemented"


def _parse_params(params_str: str) -> dict[str, Any]:
    _ = params_str
    assert False, "_parse_params not implemented"


def _reconstruct_initial_state(stand_id: str, in_db: sqlite3.Connection) -> ForestStand:
    # TODO: how to reconstruct initial state if and when original simulation db has incomplete attributes?
    #           - always complete output for initial state?
    #               - would lead to lots of mostly empty columns...
    #               - unless we add new tables specifically for the initial state?
    #           - require original source data or preprocessed csv?
    #               - how to deal with potential updating?
    _ = stand_id
    _ = in_db
    assert False, "_reconstruct_initial_state not implemented"
