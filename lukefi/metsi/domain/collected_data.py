from enum import StrEnum
import sqlite3
from typing import override

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


def init_collected_data_tables(db: sqlite3.Connection, data_types: set[CollectableData]):
    cur = db.cursor()
    for data_type in data_types:
        cur.execute(DB_TABLE_INIT[data_type])


class RemovedTrees(CollectedData):

    removed_trees: ReferenceTrees

    @override
    def output_to_db(self, db: sqlite3.Connection, node_str: str, identifier: str):
        cur = db.cursor()
        for i in range(self.removed_trees.size):
            cur.execute(
                """
                INSERT INTO removed_trees
                VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_str,
                    identifier,
                    self.removed_trees.identifier[i],
                    int(self.removed_trees.tree_number[i]),
                    int(self.removed_trees.species[i]),
                    self.removed_trees.breast_height_diameter[i],
                    self.removed_trees.height[i],
                    self.removed_trees.measured_height[i],
                    self.removed_trees.breast_height_age[i],
                    self.removed_trees.biological_age[i],
                    self.removed_trees.stems_per_ha[i],
                    int(self.removed_trees.origin[i]),
                    int(self.removed_trees.management_category[i]),
                    self.removed_trees.saw_log_volume_reduction_factor[i],
                    int(self.removed_trees.pruning_year[i]),
                    int(self.removed_trees.age_when_10cm_diameter_at_breast_height[i]),
                    f"({self.removed_trees.stand_origin_relative_position[i][0]}, "
                    f"{self.removed_trees.stand_origin_relative_position[i][1]}, "
                    f"{self.removed_trees.stand_origin_relative_position[i][2]})",
                    self.removed_trees.lowest_living_branch_height[i],
                    self.removed_trees.tree_category[i],
                    int(self.removed_trees.storey[i]),
                    bool(self.removed_trees.sapling[i]),
                    self.removed_trees.tree_type[i],
                    self.removed_trees.tuhon_ilmiasu[i]
                )
            )
