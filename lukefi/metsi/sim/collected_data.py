from abc import ABC, abstractmethod
import sqlite3
from typing import TypeVar


class CollectedData(ABC):

    @abstractmethod
    def output_to_db(self, db: sqlite3.Connection, node_str: str, identifier: str):
        pass

T = TypeVar("T")
OpTuple = tuple[T, list[CollectedData]]
