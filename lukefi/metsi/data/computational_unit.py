from abc import ABC, abstractmethod
import sqlite3


class ComputationalUnit(ABC):
    identifier: str
    time: int = 0
    start_time: int = 0

    @abstractmethod
    def output_to_db(self, db: sqlite3.Connection, node: str):
        pass

    @abstractmethod
    def update_aggregates(self):
        pass

    @property
    def relative_time(self):
        return self.time - self.start_time
