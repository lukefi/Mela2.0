from abc import ABC, abstractmethod
import sqlite3
from typing import TYPE_CHECKING, Type, TypeVar

if TYPE_CHECKING:
    from lukefi.metsi.sim.treatment import PredeterminedTreatment

T = TypeVar("T", bound='ComputationalUnit')


class ComputationalUnit(ABC):
    identifier: str
    time: int = 0
    start_time: int = 0
    predetermined_treatments: list[tuple[int, "PredeterminedTreatment"]] | None

    @abstractmethod
    def output_initial_state_to_db(self, db: sqlite3.Connection):
        pass

    @abstractmethod
    def output_to_db(self, db: sqlite3.Connection, node: str):
        pass

    @abstractmethod
    def update_aggregates(self):
        pass

    @property
    def relative_time(self):
        return self.time - self.start_time

    @classmethod
    @abstractmethod
    def reconstruct_initial_state(cls: Type[T], identifier: str, db: sqlite3.Connection) -> T:
        pass

    @classmethod
    @abstractmethod
    def create_database_tables(cls, db: sqlite3.Connection, sqlite_decl: dict[str, str] | None = None):
        pass
