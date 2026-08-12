from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar("T", bound="Finalizable")

class Finalizable(ABC):
    @abstractmethod
    def finalize(self: T) -> T:
        pass
