from abc import ABC, abstractmethod
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lukefi.metsi.sim.treatment import PredeterminedTreatment


class ComputationalUnit(ABC):
    identifier: str
    time: int = 0
    start_time: int = 0
    predetermined_treatments: list["PredeterminedTreatment"] | None

    @abstractmethod
    def output_to_db(self, db: sqlite3.Connection, node: str):
        pass

    @abstractmethod
    def update_aggregates(self):
        pass

    @property
    def relative_time(self):
        return self.time - self.start_time
