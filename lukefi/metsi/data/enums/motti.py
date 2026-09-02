from enum import IntEnum


class MottiSpecies(IntEnum):
    UNKNOWN = 0
    PINE = 1
    SPRUCE = 2
    SILVER_BIRCH = 3
    DOWNY_BIRCH = 4
    ASPEN = 5
    ALDER = 6
    OTHER_CONIFEROUS = 8
    OTHER_DECIDUOUS = 9


class MottiStorey(IntEnum):
    UNDER = 1
    DOMINANT = 2
    OVER = 3
    SPARE = 4


class MottiRegenerationMethod(IntEnum):
    NATURAL = 1
    SOWING = 2
    PLANTING = 3


class MottiDrainageCategory(IntEnum):
    # NOTE: What are these in eng?
    OJITTAMATON_KANGAS = 0
    OJITETTU_KANGAS = 1
    OJITTAMATON_SUO = 2
    OJIKKO = 3
    MUUTTUMA = 4
    TURVEKANGAS = 5

class MottiSiteType(IntEnum):
    VERY_RICH_SITE = 1
    RICH_SITE = 2
    DAMP_SITE = 3
    SUB_DRY_SITE = 4
    DRY_SITE = 5
    BARREN_SITE = 6
    HERB_RICH_TYPE = 51
    VACCINIUM_MYRTILLUS_TYPE_1 = 52
    VACCINIUM_MYRTILLUS_TYPE_2 = 53
    VACCINIUM_VITIS_IDAEA_TYPE = 54
    DEV_FROM_GENUINE_FORESTED_MIRE = 55
    DWARF_SHRUB_TYPE = 56
    CLADONIA_TYPE = 57
