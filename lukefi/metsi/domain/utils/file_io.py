import sqlite3
from typing import Optional

from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.collected_data import CollectedData
from lukefi.metsi.sim.simulation_payload import SimulationPayload

STANDS_TYPES = {
    "year": "INTEGER",
    "stand_id": "INTEGER",
    "area": "REAL",
    "area_weight": "REAL",
    "geo_location": "TEXT",
    "degree_days": "REAL",
    "owner_category": "INTEGER",
    "land_use_category": "INTEGER",
    "soil_peatland_category": "INTEGER",
    "site_type_category": "INTEGER",
    "tax_class_reduction": "INTEGER",
    "tax_class": "INTEGER",
    "drainage_category": "INTEGER",
    "drainage_year": "INTEGER",
    "fertilization_year": "INTEGER",
    "soil_surface_preparation_year": "INTEGER",
    "regeneration_area_cleaning_year": "INTEGER",
    "development_class": "INTEGER",
    "artificial_regeneration_year": "INTEGER",
    "young_stand_tending_year": "INTEGER",
    "cutting_year": "INTEGER",
    "forestry_centre_id": "INTEGER",
    "forest_management_category": "REAL",
    "method_of_last_cutting": "INTEGER",
    "municipality_id": "INTEGER",
    "ds_main_tree_species_biological_age": "REAL",
    "area_weight_factors": "TEXT",
    "fra_category": "TEXT",
    "auxiliary_stand": "INTEGER",
    "sea_effect": "REAL",
    "lake_effect": "REAL",
    "basal_area": "REAL",
    "main_tree_species_dominant_storey": "INTEGER",
    "ds_dominant_height": "REAL",
    "region": "INTEGER",
    "peatland_type": "INTEGER",
    "drained_peatland_type": "INTEGER",
    "under_storey": "INTEGER",
    "over_storey": "INTEGER",

}

TREES_TYPES = {
    "tree_number": "INTEGER",
    "species": "INTEGER",
    "breast_height_diameter": "REAL",
    "height": "REAL",
    "measured_height": "REAL",
    "breast_height_age": "REAL",
    "biological_age": "REAL",
    "stems_per_ha": "REAL",
    "origin": "INTEGER",
    "management_category": "INTEGER",
    "tree_category": "TEXT",
    "storey": "INTEGER",
    "sapling": "INTEGER",
    "tree_type": "TEXT",
    "damage_type": "TEXT",
    "basal_area": "REAL",
    "volume": "REAL",
    "stratum": "INTEGER"
}
STRATA_TYPES = {
    "species": "INTEGER",
    "mean_diameter": "REAL",
    "mean_height": "REAL",
    "breast_height_age": "REAL",
    "biological_age": "REAL",
    "stems_per_ha": "REAL",
    "basal_area": "REAL",
    "origin": "INTEGER",
    "stratum_number": "INTEGER",
    "storey": "INTEGER",
    "sapling_stems_per_ha": "REAL",
    "number_of_generated_trees": "INTEGER",
}


def _select_columns(table: str, decl: Optional[dict]) -> list[str]:
    if not decl:
        # default = all fields
        if table == "stands":
            return list(STANDS_TYPES.keys())
        if table == "trees":
            return list(TREES_TYPES.keys())
        if table == "strata":
            return list(STRATA_TYPES.keys())
        return []
    return list(decl.get(table, []))


def create_database_tables(db: sqlite3.Connection, sqlite_decl: Optional[dict] = None):
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

    # stands: required id fields + declared fields
    stand_cols = _select_columns("stands", sqlite_decl)
    # required id cols for stands table:
    stand_prefix = ["node TEXT", "identifier TEXT"]

    stand_decl = [f"{c} {STANDS_TYPES[c]}" for c in stand_cols]
    cur.execute(
        f"""--sql
        CREATE TABLE stands(
            {", ".join(stand_prefix + stand_decl)},
            PRIMARY KEY(node, identifier),
            FOREIGN KEY(node, identifier) REFERENCES nodes(identifier, stand))
        """
    )

    # trees: required id cols + declared cols
    tree_cols = _select_columns("trees", sqlite_decl)
    tree_prefix = ["node TEXT", "stand TEXT", "identifier TEXT"]
    tree_decl = [f"{c} {TREES_TYPES[c]}" for c in tree_cols]
    cur.execute(
        f"""--sql
        CREATE TABLE trees(
            {", ".join(tree_prefix + tree_decl)},
            PRIMARY KEY (node, identifier),
            FOREIGN KEY (node, stand) REFERENCES nodes(identifier, stand))
        """
    )

    # strata: required id cols + declared cols
    strata_cols = _select_columns("strata", sqlite_decl)
    strata_prefix = ["node TEXT", "stand TEXT", "identifier TEXT"]
    strata_decl = [f"{c} {STRATA_TYPES[c]}" for c in strata_cols]
    cur.execute(
        f"""--sql
        CREATE TABLE strata(
            {", ".join(strata_prefix + strata_decl)},
            PRIMARY KEY (node, identifier),
            FOREIGN KEY (node, stand) REFERENCES nodes(identifier, stand))
        """
    )


def output_node_to_db[T: ComputationalUnit](db: sqlite3.Connection,
                                            current: SimulationPayload[T],
                                            collected_data: list[CollectedData],
                                            tags: Optional[set[str]] = None,
                                            output_state: bool = True,
                                            output_collected_data: bool = True,
                                            is_transition: bool = False):
    """
    Writes current simulation state and collected data to database.

    :param db: Connection to an initialized database
    :param current: The current simulation payload (e.g. state and treatment history)
    :param collected_data: List of data collected by the treatment performed in the current node
    """
    if tags is None:
        tags = set()
    node_str = "-".join(map(str, current.node_id))
    if is_transition:
        node_str += "-T"
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
         str(tags) if len(tags) > 0 else "{}"))
    if output_state:
        current.computational_unit.output_to_db(db, node_str)
    if output_collected_data:
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
