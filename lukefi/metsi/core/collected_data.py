from abc import ABC, abstractmethod
import sqlite3

class CollectedData(ABC):
    """ Interface for data collected during simulation and stored in SQLite.

    Implementations define the database schema and how collected data is
    inserted into it. Transaction management is left to the implementation
    and the calling application.
    """

    @classmethod
    @abstractmethod
    def init_db_table(cls, db: sqlite3.Connection) -> None:
        """ Create the SQLite table and schema for this collected data type.

        Args:
            db: SQLite database connection.
        """


    @abstractmethod
    def output_to_db(self, db: sqlite3.Connection, node_str: str, identifier: str) -> None:
        """ Insert the collected data into its SQLite table.
        
            Args:
                db: SQLite database connection.
                node_str: Unique string identifying the simulation node.
                identifier: Unique string identifying the computational unit.             
        """


def init_collected_data_tables(db: sqlite3.Connection,
                               data_types: "CollectableDataTypes",
                               existing_data_types: "CollectableDataTypes | None" = None):
    for data_type in data_types:
        if existing_data_types is None or data_type not in existing_data_types:
            data_type.init_db_table(db)


type OpTuple[T] = tuple[T, list[CollectedData]]
type CollectableDataTypes = set[type[CollectedData]]
