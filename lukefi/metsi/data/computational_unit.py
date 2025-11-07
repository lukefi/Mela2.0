from abc import ABC, abstractmethod
import sqlite3


class ComputationalUnit(ABC):
    identifier: str

    @abstractmethod
    def output_to_db(self, db: sqlite3.Connection, node: str):
        pass

    @abstractmethod
    def update_aggregates(self):
        pass
