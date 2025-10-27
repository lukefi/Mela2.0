from abc import ABC, abstractmethod
import sqlite3
from typing import TypeVar


class CollectedData(ABC):

    @classmethod
    @abstractmethod
    def init_db_table(cls, db: sqlite3.Connection):
        pass

    @abstractmethod
    def output_to_db(self, db: sqlite3.Connection, node_str: str, identifier: str):
        pass


def init_collected_data_tables(db: sqlite3.Connection, data_types: "CollectableDataTypes"):
    for data_type in data_types:
        data_type.init_db_table(db)


T = TypeVar("T")
OpTuple = tuple[T, list[CollectedData]]
CollectableDataTypes = set[type[CollectedData]]
