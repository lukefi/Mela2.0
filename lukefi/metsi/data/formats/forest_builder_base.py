from abc import ABC, abstractmethod
from enum import Enum

from lukefi.metsi.data.formats.declarative_conversion import Conversion, ConversionMapper
from lukefi.metsi.domain.forestry_types import StandList


class RowKind(Enum):
    STAND = "1"
    STRATUM = "2"
    TREE = "3"


class ForestBuilder(ABC):

    @abstractmethod
    def build(self) -> StandList:
        pass


class VMIBuilder(ForestBuilder):

    stand_rows: list[dict[str, str]]
    stratum_rows: dict[str, list[dict[str, str]]]
    tree_rows: dict[str, list[dict[str, str]]]

    builder_flags: dict[str, bool]
    conversion_reader: ConversionMapper

    def __init__(self, builder_flags: dict[str, bool], declared_conversions: dict[str, Conversion]) -> None:
        self.stand_rows = []
        self.stratum_rows = {}
        self.tree_rows = {}

        self.builder_flags = builder_flags
        self.conversion_reader = ConversionMapper(declared_conversions)


class VMI9Builder(VMIBuilder):
    pass


class VMI10Builder(VMIBuilder):
    pass


class VMI11Builder(VMIBuilder):
    pass
