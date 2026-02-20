import sqlite3
from typing import Optional

from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.collected_data import CollectedData
from lukefi.metsi.sim.simulation_payload import SimulationPayload


def create_database_tables(db: sqlite3.Connection):
    cur = db.cursor()
    cur.execute(
        """--sql
        CREATE TABLE nodes(
            identifier TEXT,
            stand TEXT,
            done_treatment TEXT,
            treatment_params TEXT,
            tags TEXT,
            leaf INTEGER(1) DEFAULT(0),
            PRIMARY KEY(identifier, stand))
        """
    )
    cur.execute(
        """--sql
        CREATE TABLE stands(
            node TEXT,
            identifier TEXT,
            year INTEGER,
            stand_id INTEGER,
            area REAL,
            area_weight REAL,
            geo_location TEXT,
            degree_days REAL,
            owner_category INTEGER,
            land_use_category INTEGER,
            soil_peatland_category INTEGER,
            site_type_category INTEGER,
            tax_class_reduction INTEGER,
            tax_class INTEGER,
            drainage_category INTEGER,
            drainage_year INTEGER,
            fertilization_year INTEGER,
            soil_surface_preparation_year INTEGER,

            regeneration_area_cleaning_year INTEGER,
            development_class INTEGER,
            artificial_regeneration_year INTEGER,
            young_stand_tending_year INTEGER,
            cutting_year INTEGER,
            forestry_centre_id INTEGER,
            forest_management_category REAL,
            method_of_last_cutting INTEGER,
            municipality_id INTEGER,
            dominant_storey_age REAL,
            area_weight_factors TEXT,
            fra_category TEXT,
            land_use_category_detail TEXT,
            auxiliary_stand INTEGER(1),
            sea_effect REAL,
            lake_effect REAL,
            basal_area REAL,
            main_tree_species_dominant_storey INTEGER,
            dominant_height_dominant_storey REAL,
            region INTEGER,
            PRIMARY KEY(node, identifier),
            FOREIGN KEY(node, identifier) REFERENCES nodes(identifier, stand))
        """
    )
    cur.execute(
        """--sql
        CREATE TABLE trees(
            node TEXT,
            stand TEXT,
            identifier TEXT,
            tree_number INTEGER,
            species INTEGER,
            breast_height_diameter REAL,
            height REAL,
            measured_height REAL,
            breast_height_age REAL,
            biological_age REAL,
            stems_per_ha REAL,
            origin INTEGER,
            management_category INTEGER,
            tree_category TEXT,
            storey INTEGER,
            sapling INTEGER(1),
            tree_type TEXT,
            tuhon_ilmiasu TEXT,
            basal_area REAL,
            volume REAL,
            PRIMARY KEY (node, identifier),
            FOREIGN KEY (node, stand) REFERENCES nodes(identifier, stand))
        """
    )
    cur.execute(
        """--sql
        CREATE TABLE strata(
            node TEXT,
            stand TEXT,
            identifier TEXT,
            species INTEGER,
            mean_diameter REAL,
            mean_height REAL,
            breast_height_age REAL,
            biological_age REAL,
            stems_per_ha REAL,
            basal_area REAL,
            origin INTEGER,
            tree_number INTEGER,
            storey INTEGER,
            sapling_stems_per_ha REAL,
            number_of_generated_trees INTEGER,
            PRIMARY KEY (node, identifier),
            FOREIGN KEY (node, stand) REFERENCES nodes(identifier, stand))
        """
    )


def output_node_to_db[T: ComputationalUnit](db: sqlite3.Connection,
                                            current: SimulationPayload[T],
                                            collected_data: list[CollectedData],
                                            tags: Optional[set[str]] = None):
    """
    Writes current simulation state and collected data to database.

    :param db: Connection to an initialized database
    :param current: The current simulation payload (e.g. state and treatment history)
    :param collected_data: List of data collected by the treatment performed in the current node
    """
    if tags is None:
        tags = set()
    node_str = "-".join(map(str, current.node_id))
    cur = db.cursor()
    cur.execute(
        """--sql
        INSERT INTO nodes (identifier, stand, done_treatment, treatment_params, tags)
        VALUES
            (?, ?, ?, ?, ?)
        """,
        (node_str,
         current.computational_unit.identifier,
         current.operation_history[-1][1] if len(current.operation_history) > 0 else "do_nothing",
         str(current.operation_history[-1][2]) if len(current.operation_history) > 0 else "{}",
         str(tags)))
    current.computational_unit.output_to_db(db, node_str)
    for datum in collected_data:
        datum.output_to_db(db, node_str, current.computational_unit.identifier)


def update_leaf_node[T: ComputationalUnit](db: sqlite3.Connection, leaf_node: SimulationPayload[T]):
    cur = db.cursor()
    cur.execute(
        """--sql
        UPDATE nodes
        SET leaf = 1
        WHERE
            identifier = ? AND
            stand = ?;
        """,
        ("-".join(map(str, leaf_node.node_id)),
            leaf_node.computational_unit.identifier)
    )
