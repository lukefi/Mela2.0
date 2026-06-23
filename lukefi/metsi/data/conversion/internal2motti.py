from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.enums.internal import (
    CONIFEROUS_SPECIES, DECIDUOUS_SPECIES, DrainageCategory, Storey, TreeSpecies)
from lukefi.metsi.data.enums.motti import MottiSpecies, MottiStorey, MottiDrainageCategory


_SPECIES_MAP = {
    TreeSpecies.PINE: MottiSpecies.PINE,
    TreeSpecies.SPRUCE: MottiSpecies.SPRUCE,
    TreeSpecies.SILVER_BIRCH: MottiSpecies.SILVER_BIRCH,
    TreeSpecies.DOWNY_BIRCH: MottiSpecies.DOWNY_BIRCH,
    TreeSpecies.ASPEN: MottiSpecies.ASPEN,
    TreeSpecies.GREY_ALDER: MottiSpecies.ALDER,
    TreeSpecies.COMMON_ALDER: MottiSpecies.ALDER,
    TreeSpecies.UNKNOWN: MottiSpecies.UNKNOWN
}


_STOREY_MAP = {
    Storey.DOMINANT: MottiStorey.DOMINANT,
    Storey.UNDER: MottiStorey.UNDER,
    Storey.OVER: MottiStorey.OVER,
    Storey.SPARE: MottiStorey.SPARE,
}

_DRAINAGE_CATEGORY_MAP = {
    DrainageCategory.UNDRAINED_MINERAL_SOIL_OR_MIRE: MottiDrainageCategory.OJITTAMATON_KANGAS,
    DrainageCategory.UNDRAINED_MINERAL_SOIL: MottiDrainageCategory.OJITTAMATON_KANGAS,
    DrainageCategory.MINERAL_SOIL_TURNED_MIRE: MottiDrainageCategory.OJITETTU_KANGAS,
    DrainageCategory.DITCHED_MINERAL_SOIL: MottiDrainageCategory.OJITETTU_KANGAS,
    DrainageCategory.UNDRAINED_MIRE: MottiDrainageCategory.OJITTAMATON_SUO,
    DrainageCategory.DITCHED_MIRE: MottiDrainageCategory.OJIKKO,
    DrainageCategory.TRANSFORMING_MIRE: MottiDrainageCategory.MUUTTUMA,
    DrainageCategory.TRANSFORMED_MIRE: MottiDrainageCategory.TURVEKANGAS
}


def convert_species(source: TreeSpecies) -> MottiSpecies:
    """
    Map internal TreeSpecies -> Motti species codes directly.
    - Keep main species 1..5 as-is
    - Collapse both alders (GREY_ALDER, COMMON_ALDER) to 6
    - If in CONIFEROUS_SPECIES -> 8
    - If in DECIDUOUS_SPECIES -> 9
    """
    if source in _SPECIES_MAP:
        return _SPECIES_MAP[source]
    if source in CONIFEROUS_SPECIES:
        return MottiSpecies.OTHER_CONIFEROUS
    if source in DECIDUOUS_SPECIES:
        return MottiSpecies.OTHER_DECIDUOUS

    raise MetsiException(f"Unable to map internal species {source} to Motti species")


def convert_drainage_category(source: DrainageCategory | None) ->  MottiDrainageCategory:
    """ Drainage category transformation from internal to motti value.
     
     defaults undrained mineral soil which is valued as zero. 
     """
    if source is None:
        return MottiDrainageCategory.OJITTAMATON_KANGAS
    if source in _DRAINAGE_CATEGORY_MAP:
        return _DRAINAGE_CATEGORY_MAP[source]
    return MottiDrainageCategory.OJITTAMATON_KANGAS
