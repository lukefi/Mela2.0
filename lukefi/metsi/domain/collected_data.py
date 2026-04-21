import sqlite3
from typing import override

from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.sim.collected_data import CollectedData


class RemovedTrees(CollectedData):

    removed_trees: ReferenceTrees

    @classmethod
    @override
    def init_db_table(cls, db: sqlite3.Connection):
        cur = db.cursor()
        cur.execute("""--sql
            CREATE TABLE removed_trees(
                node TEXT,
                stand TEXT,
                identifier TEXT,
                tree_number INTEGER,
                species INTEGER,
                breast_height_diameter REAL,
                height REAL,
                stems_per_ha REAL,
                origin INTEGER,
                breast_height_age REAL,
                volume REAL,
                PRIMARY KEY (node, identifier),
                FOREIGN KEY (node, stand) REFERENCES nodes(identifier, stand)
            )
        """)

    @override
    def output_to_db(self, db: sqlite3.Connection, node_str: str, identifier: str):
        cur = db.cursor()
        for i in range(self.removed_trees.size):
            cur.execute(
                """--sql
                INSERT INTO removed_trees
                VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_str,
                    identifier,
                    self.removed_trees.identifier[i],
                    int(self.removed_trees.tree_number[i]),
                    int(self.removed_trees.species[i]),
                    self.removed_trees.breast_height_diameter[i],
                    self.removed_trees.height[i],
                    self.removed_trees.stems_per_ha[i],
                    int(self.removed_trees.origin[i]),
                    self.removed_trees.breast_height_age[i],
                    self.removed_trees.volume[i]
                )
            )
