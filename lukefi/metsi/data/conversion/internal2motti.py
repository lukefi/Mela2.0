from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.enums.internal import CONIFEROUS_SPECIES, DECIDUOUS_SPECIES, Storey, TreeSpecies
from lukefi.metsi.data.enums.motti import MottiSpecies, MottiStorey


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
