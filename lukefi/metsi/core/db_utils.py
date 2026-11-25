from enum import IntEnum
import sqlite3
from typing import Any, Optional

from lukefi.metsi.core.collected_data import CollectedData
from lukefi.metsi.core.model import ComputationalUnit
from lukefi.metsi.core.simulation_payload import SimulationPayload


class NodeType(IntEnum):
    """ Types of nodes used for categorization """
    INITIAL = 0
    TREATMENT = 1
    TRANSITION = 2
    TREATMENT_LEAF = 3
    TRANSITION_LEAF = 4
    UPDATING_TREATMENT = 5
    UPDATING_TRANSITION = 6
    RESIMULATION_TREATMENT = 7
    RESIMULATION_TRANSITION = 8
    RESIMULATION_TREATMENT_LEAF = 9
    RESIMULATION_TRANSITION_LEAF = 10


def create_database_tables(
        db: sqlite3.Connection,
        unit_type: type[ComputationalUnit],
        sqlite_decl: Optional[dict] = None):
    """ Creates node and computational unit related tables into SQLite database.
    
    :param db: Database connection.
    :param unit_type: Type of computational unit for creating output data tables.
    :param sqlite_decl: (Optional) Schema for computational unit output tables.
    """
    cur = db.cursor()

    # nodes
    cur.execute(
        """--sql
        CREATE TABLE nodes(
            identifier TEXT,
            unit TEXT,
            done_treatment TEXT,
            treatment_params TEXT,
            tags TEXT,
            node_type INTEGER DEFAULT(0),
            PRIMARY KEY(identifier, unit))
        """
    )

    # computationan unit specific
    unit_type.create_database_tables(db, sqlite_decl)


def output_node_to_db(db: sqlite3.Connection,
                      node_id: list[int],
                      operation: str,
                      params: dict[str, Any],
                      unit: ComputationalUnit,
                      collected_data: list[CollectedData],
                      tags: Optional[set[str]] = None,
                      output_state: bool = True,
                      output_cd: bool = True,
                      transition_count: int = 0,
                      node_type: NodeType = NodeType.TREATMENT,
                      ):
    """
    Writes current simulation state and collected data to database.

    :param db: Connection to an initialized database
    :param node_id: Identifier/position of the current node in branch.
    :param operation: Name of the operation performed for current node.
    :param params: Mapping of parameters that the operation used.
    :param unit: Computational unit beholding the domain-state of the current node. 
    :param collected_data: List of data collected by the treatment performed in the current node.
    :param tags: (Optional) Set of alias names for operation.
    :param output_state: Boolean flag for outputting the current state.
    :param output_cd: Boolean flag for outputting the data operation collected.
    :param transition_count: Number of transition calls.
    :param node_type: Categorical type value for current node.
    """
    if tags is None:
        tags = set()
    node_str = "-".join(map(str, node_id))
    if transition_count:
        node_str += "-T" * transition_count

    cur = db.cursor()
    cur.execute(
        """--sql
        INSERT INTO nodes (identifier, unit, done_treatment, treatment_params, tags, node_type)
        VALUES
            (?, ?, ?, ?, ?, ?)
        """,
        (node_str,
         unit.identifier,
         operation,
         str(params),
         str(tags) if len(tags) > 0 else "{}",
         node_type))
    if output_state:
        unit.output_to_db(db, node_str)
    if output_cd:
        for datum in collected_data:
            datum.output_to_db(db, node_str, unit.identifier)


def update_leaf_node[T: ComputationalUnit](
        db: sqlite3.Connection,
        leaf_node: SimulationPayload[T],
        transition_count: int):
    """ Updates given node type to treatment or transition leaf at the end of simulation """
    cur = db.cursor()
    node_id = "-".join(map(str, leaf_node.node_id))
    cur.execute(
        """--sql
        UPDATE nodes
        SET node_type = ?
        WHERE
            identifier = ?
            AND unit = ?;
        """,
        (
            NodeType.TREATMENT_LEAF,
            node_id,
            leaf_node.unit.identifier
        )
    )
    cur.execute(
        """--sql
        UPDATE nodes
        SET node_type = ?
        WHERE
            identifier = ?
            AND unit = ?;
        """,
        (
            NodeType.TRANSITION_LEAF,
            node_id + ("-T" * transition_count),
            leaf_node.unit.identifier
        )
    )
