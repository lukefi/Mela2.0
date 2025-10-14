from enum import StrEnum
import sqlite3

from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.sim.collected_data import CollectedData


class CollectableData(StrEnum):
    REMOVED_TREES = "removed_trees"


DB_TABLE_INIT = {
    CollectableData.REMOVED_TREES: """
        CREATE TABLE removed_trees(
            node, stand, identifier, tree_number, species, breast_height_diameter, height,
            measured_height, breast_height_age, biological_age, stems_per_ha, origin,
            management_category, saw_log_volume_reduction_factor, pruning_year,
            age_when_10cm_diameter_at_breast_height, stand_origin_relative_position,
            lowest_living_branch_height, tree_category, storey, sapling, tree_type, tuhon_ilmiasu,
            PRIMARY KEY (node, identifier),
            FOREIGN KEY (node, stand) REFERENCES nodes(identifier, stand)
        )
    """
}


class RemovedTrees(CollectedData):

    removed_trees: ReferenceTrees
    
    def output_to_db(self, db: sqlite3.Connection, node_str: str, identifier: str):
        pass
