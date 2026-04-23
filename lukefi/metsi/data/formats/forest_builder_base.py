from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from lukefi.metsi.data.conversion import vmi2internal
from lukefi.metsi.data.formats import util, vmi_util
from lukefi.metsi.data.formats.declarative_conversion import ConversionMapper
from lukefi.metsi.data.model import ForestStand
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

    def __init__(self, builder_flags: dict[str, bool]) -> None:
        self.stand_rows = []
        self.stratum_rows = {}
        self.tree_rows = {}

        self.builder_flags = builder_flags


class VMI9Builder(VMIBuilder):
    pass


class VMI10Builder(VMIBuilder):
    pass


class VMI11Builder(VMIBuilder):
    pass


class VMI12Builder(VMIBuilder):
    pass


class ForestCentreBuilder(ForestBuilder):
    ''' Base class for building a forest data model from Forest Centre (Suomen Metsakeskus) source '''

    @abstractmethod
    def build(self) -> StandList:
        ...

    @abstractmethod
    def convert_stand_entry(self, entry) -> ForestStand:
        ...


class XMLBuilder(ForestCentreBuilder):
    pass


class GeoPackageBuilder(ForestCentreBuilder):
    pass
