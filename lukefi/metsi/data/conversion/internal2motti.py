from lukefi.metsi.core.exceptions import MetsiException
from lukefi.metsi.data.enums.internal import (
    MetsiEnum, CONIFEROUS_SPECIES, DECIDUOUS_SPECIES,
    DrainageCategory, DrainedPeatlandForestType, Storey, TreeSpecies, SiteType)
from lukefi.metsi.data.enums.motti import (
    MottiSpecies, MottiStorey, MottiDrainageCategory, MottiSiteType)




_COMMON_SITE_TYPES = [
    # Common site type values (same are used also for drained peatlands).
    MottiSiteType.VERY_RICH_SITE,
    MottiSiteType.RICH_SITE,
    MottiSiteType.DAMP_SITE,
    MottiSiteType.SUB_DRY_SITE,
    MottiSiteType.DRY_SITE,
    MottiSiteType.BARREN_SITE
]


_DRAINED_PEATLAND_SITE_TYPE_SPESIFICATIONS = [
    # Drained peatland forest type spesification values (from 51-57).
    MottiSiteType.HERB_RICH_TYPE,
    MottiSiteType.VACCINIUM_MYRTILLUS_TYPE_1,
    MottiSiteType.VACCINIUM_MYRTILLUS_TYPE_2,
    MottiSiteType.VACCINIUM_VITIS_IDAEA_TYPE,
    MottiSiteType.DEV_FROM_GENUINE_FORESTED_MIRE,
    MottiSiteType.DWARF_SHRUB_TYPE,
    MottiSiteType.CLADONIA_TYPE
]



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

_SITE_TYPE_MAP: dict[MetsiEnum | None, MottiSiteType] = {
    SiteType.VERY_RICH_SITE: MottiSiteType.VERY_RICH_SITE,
    SiteType.RICH_SITE: MottiSiteType.RICH_SITE,
    SiteType.DAMP_SITE: MottiSiteType.DAMP_SITE,
    SiteType.SUB_DRY_SITE: MottiSiteType.SUB_DRY_SITE,
    SiteType.DRY_SITE: MottiSiteType.DRY_SITE,
    SiteType.BARREN_SITE: MottiSiteType.BARREN_SITE
}

_DRAINED_PEATLAND_FOREST_TYPE_MAP: dict[MetsiEnum | None, MottiSiteType] = {
    DrainedPeatlandForestType.HERB_RICH_TYPE: MottiSiteType.HERB_RICH_TYPE,
    DrainedPeatlandForestType.VACCINIUM_MYRTILLUS_TYPE_1: MottiSiteType.VACCINIUM_MYRTILLUS_TYPE_1,
    DrainedPeatlandForestType.VACCINIUM_MYRTILLUS_TYPE_2: MottiSiteType.VACCINIUM_MYRTILLUS_TYPE_2,
    DrainedPeatlandForestType.VACCINIUM_VITIS_IDAEA_TYPE: MottiSiteType.VACCINIUM_VITIS_IDAEA_TYPE,
    DrainedPeatlandForestType.DEV_FROM_GENUINE_FORESTED_MIRE: MottiSiteType.DEV_FROM_GENUINE_FORESTED_MIRE,
    DrainedPeatlandForestType.DWARF_SHRUB_TYPE: MottiSiteType.DWARF_SHRUB_TYPE,
    DrainedPeatlandForestType.CLADONIA_TYPE: MottiSiteType.CLADONIA_TYPE
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
    """
    Drainage category transformation from internal to motti value.
     
    defaults undrained mineral soil which is valued as zero. 
     """
    if source is None:
        return MottiDrainageCategory.OJITTAMATON_KANGAS
    if source in _DRAINAGE_CATEGORY_MAP:
        return _DRAINAGE_CATEGORY_MAP[source]
    return MottiDrainageCategory.OJITTAMATON_KANGAS


def resolve_site_type(source1: DrainedPeatlandForestType | None, source2: SiteType | None) -> MottiSiteType:
    """
    Resolves Motti site type (kasvupaikka) transformation based on internal presentation of the
    drained peatland type or the actual site type variable.
    """
    # First tries to resolve with the drained peatland type value
    if source1 is not None and source1 in _DRAINED_PEATLAND_SITE_TYPE_SPESIFICATIONS:
        return _DRAINED_PEATLAND_FOREST_TYPE_MAP[source1]

    # Fallback to resolve from common site types
    if source2 is not None and source2 in _COMMON_SITE_TYPES:
        return _SITE_TYPE_MAP[source2]

    # Raise a informative error
    _valid_values = [ str(enum.value)
                     for enum in _COMMON_SITE_TYPES + _DRAINED_PEATLAND_SITE_TYPE_SPESIFICATIONS ]
    raise MetsiException(f"Unable to resolve internal site type value [{ source2 }] \
                         or drained peatland spesific value [{ source1 }] for Motti site type value. \
                         Correct values for Motti site type are: { _valid_values }")
