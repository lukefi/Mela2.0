import ast
from copy import deepcopy
import csv
import sqlite3
from typing import Any, Set
import numpy as np

from lukefi.metsi.data.enums.internal import (
    CuttingMethod,
    DevelopmentClass,
    DrainageCategory,
    DrainedPeatlandForestType,
    FraLandUseClass,
    OwnerCategory,
    PeatlandForestType,
    SiteType,
    SoilPeatlandCategory)
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.domain.utils.file_io import NodeType, output_node_to_db
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
    # TODO: exact format of schedules csv not fully specified
    schedules_file_path = control["selected_schedules_file"]

    # TODO: recreating dynamic parameters and other complex structures from original control file
    #           - possible quick hack - declare LUT in resim control?
    #           - no need initially - dynamic parameters needed only in Monte-Carlo
    treatment_map = control["treatment_map"]

    transition = _determine_transition(control, in_db)
    instructions_per_stand = _recreate_instructions(schedules_file_path, treatment_map, in_db)

    for stand, instructions in instructions_per_stand.items():
        print(f"Resimulating stand {stand}, start year {instructions.initial_state.time}")
        for i, schedule in enumerate(instructions.schedules):
            print(f"Schedule {i}:")
            current = SimulationPayload(deepcopy(instructions.initial_state))
            current.node_id[0] = i
            current.computational_unit.predetermined_treatments = schedule
            target_time = max([time for time, _ in schedule])
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
                        output_collected_data=False,
                        transition_count=1,
                        node_type=NodeType.RESIMULATION_TRANSITION
                    )
                else:
                    should_run = False

def _determine_transition(control: dict[str, Any], in_db: sqlite3.Connection) -> TransitionFn[ForestStand]:
    _ = in_db
    return control["transition"]


def _recreate_instructions(sched_file_path: str,
                           treatment_map: dict[str, TreatmentFn],
                           in_db: sqlite3.Connection) -> dict[str, ResimulationInstructions]:
    # TODO: building simulation instructions (or equivalent) from declared leaf nodes/schedules
    instructions_per_stand: dict[str, ResimulationInstructions] = {}
    with open(sched_file_path, "r", encoding="utf-8") as sched_file:
        sched_reader = csv.reader(sched_file, delimiter=",")
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
    print(f"Recreating schedule for {stand_id}: {leaf_node_id}")
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
        print(f"{step['year']}: {step['done_treatment']} ({step['tags']})")
    return stand_id, retval


def _parse_schedule_row(schedule_row: list[str]) -> tuple[str, str]:
    return schedule_row[0], schedule_row[1]


def _parse_tags(tags_str: str) -> Set[str]:
    return ast.literal_eval(tags_str)


def _parse_params(params_str: str) -> dict[str, Any]:
    return ast.literal_eval(params_str)


def _reconstruct_initial_state(stand_id: str, in_db: sqlite3.Connection) -> ForestStand:
    cur = in_db.cursor()
    cur.row_factory = sqlite3.Row
    cur.execute(
        """--sql
            SELECT * FROM initial_stands
            WHERE
                identifier = ?;
        """,
        (
            stand_id,
        )
    )
    stand_row = cur.fetchone()

    cur.row_factory = None
    cur.execute(
        """--sql
            SELECT COUNT(*) FROM initial_trees
            WHERE
                stand = ?;
        """,
        (
            stand_id,
        )
    )
    tree_count = cur.fetchone()[0]
    trees = ReferenceTrees()
    trees.size = tree_count

    trees.identifier = np.array(_fetch_initial_trees_col(stand_id, "identifier", cur), dtype=np.dtype("U30"))
    trees.tree_number = np.array(_fetch_initial_trees_col(stand_id, "tree_number", cur), dtype=np.int32)
    trees.species = np.array(_fetch_initial_trees_col(stand_id, "species", cur), dtype=np.int32)
    trees.breast_height_diameter = np.array(_fetch_initial_trees_col(
        stand_id, "breast_height_diameter", cur), dtype=np.float64)
    trees.height = np.array(_fetch_initial_trees_col(stand_id, "height", cur), dtype=np.float64)
    trees.measured_height = np.array(_fetch_initial_trees_col(stand_id, "measured_height", cur), dtype=np.float64)
    trees.breast_height_age = np.array(_fetch_initial_trees_col(stand_id, "breast_height_age", cur), dtype=np.float64)
    trees.biological_age = np.array(_fetch_initial_trees_col(stand_id, "biological_age", cur), dtype=np.float64)
    trees.stems_per_ha = np.array(_fetch_initial_trees_col(stand_id, "stems_per_ha", cur), dtype=np.float64)
    trees.origin = np.array(_fetch_initial_trees_col(stand_id, "origin", cur), dtype=np.int32)
    trees.management_category = np.array(_fetch_initial_trees_col(stand_id, "management_category", cur), dtype=np.int32)
    trees.tree_category = np.array(_fetch_initial_trees_col(stand_id, "tree_category", cur), dtype=np.dtype("U1"))
    trees.storey = np.array(_fetch_initial_trees_col(stand_id, "storey", cur), dtype=np.int32)
    trees.sapling = np.array(_fetch_initial_trees_col(stand_id, "sapling", cur), dtype=np.bool_)
    trees.tree_type = np.array(_fetch_initial_trees_col(stand_id, "tree_type", cur), dtype=np.dtype("U1"))
    trees.damage_type = np.array(_fetch_initial_trees_col(stand_id, "damage_type", cur), dtype=np.dtype("U2"))
    trees.crown_class = np.array(_fetch_initial_trees_col(stand_id, "crown_class", cur), dtype=np.dtype("U1"))
    trees.basal_area = np.array(_fetch_initial_trees_col(stand_id, "basal_area", cur), dtype=np.float64)
    trees.volume = np.array(_fetch_initial_trees_col(stand_id, "volume", cur), dtype=np.float64)
    trees.stratum = np.array(_fetch_initial_trees_col(stand_id, "stratum", cur), dtype=np.int32)

    assert len(trees.identifier) == trees.size
    assert len(trees.tree_number) == trees.size
    assert len(trees.species) == trees.size
    assert len(trees.breast_height_diameter) == trees.size
    assert len(trees.height) == trees.size
    assert len(trees.measured_height) == trees.size
    assert len(trees.breast_height_age) == trees.size
    assert len(trees.biological_age) == trees.size
    assert len(trees.stems_per_ha) == trees.size
    assert len(trees.origin) == trees.size
    assert len(trees.management_category) == trees.size
    assert len(trees.tree_category) == trees.size
    assert len(trees.storey) == trees.size
    assert len(trees.sapling) == trees.size
    assert len(trees.tree_type) == trees.size
    assert len(trees.damage_type) == trees.size
    assert len(trees.crown_class) == trees.size
    assert len(trees.basal_area) == trees.size
    assert len(trees.volume) == trees.size
    assert len(trees.stratum) == trees.size

    retval = ForestStand(
        reference_trees=trees,
        tree_strata=TreeStrata(),
        motti_state=None,
        time=stand_row["year"],
        start_time=stand_row["year"],
        identifier=stand_row["identifier"],
        stand_id=stand_row["stand_id"],
        area=stand_row["area"],
        area_weight=stand_row["area_weight"],
        geo_location=_parse_geo_location(stand_row["geo_location"]),
        degree_days=stand_row["degree_days"],
        owner_category=OwnerCategory(stand_row["owner_category"]),
        soil_peatland_category=SoilPeatlandCategory(stand_row["soil_peatland_category"]),
        site_type_category=SiteType(stand_row["site_type_category"]),
        tax_class_reduction=stand_row["tax_class_reduction"],
        tax_class=stand_row["tax_class"],
        drainage_category=DrainageCategory(stand_row["drainage_category"]),
        drainage_year=stand_row["drainage_year"],
        fertilization_year=stand_row["fertilization_year"],
        soil_surface_preparation_year=stand_row["soil_surface_preparation_year"],
        regeneration_area_cleaning_year=stand_row["regeneration_area_cleaning_year"],
        development_class=DevelopmentClass(stand_row["development_class"]),
        artificial_regeneration_year=stand_row["artificial_regeneration_year"],
        young_stand_tending_year=stand_row["young_stand_tending_year"],
        cutting_year=stand_row["cutting_year"],
        forestry_centre_id=stand_row["forestry_centre_id"],
        forest_management_category=stand_row["forest_management_category"],
        method_of_last_cutting=CuttingMethod(stand_row["method_of_last_cutting"]),
        municipality_id=stand_row["municipality_id"],
        ds_main_tree_species_biological_age=stand_row["ds_main_tree_species_biological_age"],
        area_weight_factors=stand_row["area_weight_factors"],
        fra_category=FraLandUseClass(stand_row["fra_category"]),
        auxiliary_stand=bool(stand_row["auxiliary_stand"]),
        sea_effect=stand_row["sea_effect"],
        lake_effect=stand_row["lake_effect"],
        basal_area=stand_row["basal_area"],
        main_tree_species_dominant_storey=stand_row["main_tree_species_dominant_storey"],
        ds_dominant_height=stand_row["ds_dominant_height"],
        region=stand_row["region"],
        peatland_type=PeatlandForestType(stand_row["peatland_type"]) if stand_row["peatland_type"] is not None else None,
        drained_peatland_type=(DrainedPeatlandForestType(stand_row["drained_peatland_type"])
            if stand_row["drained_peatland_type"] is not None else None),
        under_storey=bool(stand_row["under_storey"]),
        over_storey=bool(stand_row["over_storey"])
    )

    return retval


def _parse_geo_location(src: str) -> tuple[float | None, float | None, float | None, str | None] | None:
    return ast.literal_eval(src)


def _fetch_initial_trees_col(stand: str, col: str, cur: sqlite3.Cursor) -> list[Any]:
    cur.execute(
        f"""--sql
            SELECT {col} FROM initial_trees
            WHERE
                stand = ?;
        """,
        (
            stand,
        )
    )
    return [row[0] for row in cur.fetchall()]
