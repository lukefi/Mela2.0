import sqlite3
from typing import override

import numpy as np

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


class NaturalProcessInfo(CollectedData):
    """
    Aggregated natural process information for gross growth calculations.
    """

    start_year: int
    step: int

    trees_before: ReferenceTrees
    trees_after: ReferenceTrees

    @classmethod
    @override
    def init_db_table(cls, db: sqlite3.Connection):
        cur = db.cursor()
        cur.execute(
            """--sql
            CREATE TABLE natural_process_info(
                node TEXT,
                stand TEXT,
                start_year INTEGER,
                step INTEGER,
                identifier TEXT,
                species INTEGER,
                stems_per_ha_before REAL,
                breast_height_diameter_before REAL,
                height_before REAL,
                stems_per_ha_after REAL,
                breast_height_diameter_after REAL,
                height_after REAL,
                PRIMARY KEY(node, identifier)
            )
            """
        )

    @override
    def output_to_db(self, db: sqlite3.Connection, node_str: str, identifier: str):
        before = self.trees_before
        cur = db.cursor()
        cur.executemany(
            """--sql
            INSERT INTO natural_process_info(
                node,
                stand,
                start_year,
                step,
                identifier,
                species,
                stems_per_ha_before,
                breast_height_diameter_before,
                height_before,
                stems_per_ha_after,
                breast_height_diameter_after,
                height_after
            )
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    node_str,
                    identifier,
                    self.start_year,
                    self.step,
                    before.identifier[i],
                    int(before.species[i]),
                    before.stems_per_ha[i],
                    before.breast_height_diameter[i],
                    before.height[i],
                    self._get_stems_per_ha_after(i),
                    self._get_breast_height_diameter_after(i),
                    self._get_height_after(i)
                ) for i in range(len(self.trees_before))
            )
        )

    def _get_stems_per_ha_after(self, index: int) -> float:
        before = self.trees_before
        after = self.trees_after

        identifier = before.identifier[index]

        # Try to find the same tree in final state
        for i in range(len(after)):
            if identifier == after.identifier[i]:
                return after.stems_per_ha[i]

        # Tree not found, try to find trees with same stratum number
        split_mask = after.stratum == before.stratum[index]
        return np.sum(after.stems_per_ha[split_mask])

    def _get_breast_height_diameter_after(self, index: int) -> float:
        before = self.trees_before
        after = self.trees_after

        identifier = before.identifier[index]
        for i in range(len(after)):
            if identifier == after.identifier[i]:
                return after.breast_height_diameter[i]

        # TODO: Implement ba-weighted average of split tree diameters
        return 0.0

    def _get_height_after(self, index: int) -> float:
        before = self.trees_before
        after = self.trees_after

        identifier = before.identifier[index]
        for i in range(len(after)):
            if identifier == after.identifier[i]:
                return after.height[i]

        # TODO: Implement ba-weighted average of split tree heights
        return 0.0
