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
