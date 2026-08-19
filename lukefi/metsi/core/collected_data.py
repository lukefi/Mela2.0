from abc import ABC, abstractmethod
import sqlite3


class CollectedData(ABC):

    @classmethod
    @abstractmethod
    def init_db_table(cls, db: sqlite3.Connection):
        pass

    @abstractmethod
    def output_to_db(self, db: sqlite3.Connection, node_str: str, identifier: str):
        pass


def init_collected_data_tables(db: sqlite3.Connection,
                               data_types: "CollectableDataTypes",
                               existing_data_types: "CollectableDataTypes | None" = None):
    for data_type in data_types:
        if existing_data_types is None or data_type not in existing_data_types:
            data_type.init_db_table(db)


type OpTuple[T] = tuple[T, list[CollectedData]]
type CollectableDataTypes = set[type[CollectedData]]
