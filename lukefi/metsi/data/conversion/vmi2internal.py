from typing import Optional

from lukefi.metsi.data.enums.vmi import (
    VmiOrigin,
    VmiSiteType,
    VmiOwnerCategory,
    VmiSoilPeatlandCategory,
    VmiSpecies,
    VmiLandUseCategory,
    VmiTreeCategory,
    VmiDrainageCategory, VmiStratumRank, VmiTreeStorey,
    VmiDrainedPeatlandForestType, VmiPeatlandForestType
)
from lukefi.metsi.data.enums.internal import (
    Origin,
    SiteType,
    OwnerCategory,
    SoilPeatlandCategory,
    TreeSpecies,
    LandUseCategory,
    DrainageCategory, Storey,
    TreeCategory,
    PeatlandForestType,
    DrainedPeatlandForestType
)

_species_map = {
    VmiSpecies.PINE: TreeSpecies.PINE,
    VmiSpecies.SPRUCE: TreeSpecies.SPRUCE,
    VmiSpecies.SILVER_BIRCH: TreeSpecies.SILVER_BIRCH,
    VmiSpecies.DOWNY_BIRCH: TreeSpecies.DOWNY_BIRCH,
    VmiSpecies.ASPEN: TreeSpecies.ASPEN,
    VmiSpecies.GREY_ALDER: TreeSpecies.GREY_ALDER,
    VmiSpecies.COMMON_ALDER: TreeSpecies.COMMON_ALDER,
    VmiSpecies.MOUNTAIN_ASH: TreeSpecies.MOUNTAIN_ASH,
    VmiSpecies.GOAT_WILLOW: TreeSpecies.GOAT_WILLOW,
    VmiSpecies.OTHER_CONIFEROUS: TreeSpecies.OTHER_CONIFEROUS,
    VmiSpecies.SHORE_PINE: TreeSpecies.SHORE_PINE,
    VmiSpecies.KEDAR: TreeSpecies.KEDAR,
    VmiSpecies.OTHER_PINE: TreeSpecies.OTHER_PINE,
    VmiSpecies.LARCH: TreeSpecies.LARCH,
    VmiSpecies.ABIES: TreeSpecies.ABIES,
    VmiSpecies.OTHER_SPRUCE: TreeSpecies.OTHER_SPRUCE,
    VmiSpecies.THUJA: TreeSpecies.THUJA,
    VmiSpecies.JUNIPER: TreeSpecies.JUNIPER,
    VmiSpecies.YEW: TreeSpecies.YEW,
    VmiSpecies.OTHER_DECIDUOUS: TreeSpecies.OTHER_DECIDUOUS,
    VmiSpecies.BAY_WILLOW: TreeSpecies.BAY_WILLOW,
    VmiSpecies.EUROPEAN_WHITE_ELM: TreeSpecies.EUROPEAN_WHITE_ELM,
    VmiSpecies.WYCH_ELM: TreeSpecies.WYCH_ELM,
    VmiSpecies.SMALL_LEAVED_LIME: TreeSpecies.SMALL_LEAVED_LIME,
    VmiSpecies.POPLAR: TreeSpecies.POPLAR,
    VmiSpecies.COMMON_ASH: TreeSpecies.COMMON_ASH,
    VmiSpecies.OAK: TreeSpecies.OAK,
    VmiSpecies.BIRD_CHERRY: TreeSpecies.BIRD_CHERRY,
    VmiSpecies.MAPLE: TreeSpecies.MAPLE,
    VmiSpecies.HAZEL: TreeSpecies.HAZEL,
    VmiSpecies.UNKNOWN: TreeSpecies.UNKNOWN,
    VmiSpecies.TREELESS: TreeSpecies.TREELESS
}


_land_use_map = {
    VmiLandUseCategory.FOREST: LandUseCategory.FOREST,
    VmiLandUseCategory.SCRUB_LAND: LandUseCategory.SCRUB_LAND,
    VmiLandUseCategory.WASTE_LAND: LandUseCategory.WASTE_LAND,
    VmiLandUseCategory.OTHER_FOREST: LandUseCategory.OTHER_FOREST,
    VmiLandUseCategory.AGRICULTURAL: LandUseCategory.AGRICULTURAL,
    VmiLandUseCategory.BUILT_LAND: LandUseCategory.BUILT_LAND,
    VmiLandUseCategory.ROAD: LandUseCategory.ROAD,
    VmiLandUseCategory.ENERGY_TRANSMISSION_LINE: LandUseCategory.ENERGY_TRANSMISSION_LINE,
    VmiLandUseCategory.FRESHWATER: LandUseCategory.FRESHWATER,
    VmiLandUseCategory.SEA: LandUseCategory.SEA,
    VmiLandUseCategory.OBSOLETE: LandUseCategory.OTHER_FOREST
}


_owner_map = {
    VmiOwnerCategory.UNKNOWN: OwnerCategory.UNKNOWN,
    VmiOwnerCategory.PRIVATE: OwnerCategory.PRIVATE,
    VmiOwnerCategory.FOREST_INDUSTRY_ENTERPRISE: OwnerCategory.FOREST_INDUSTRY,
    VmiOwnerCategory.OTHER_ENTERPRISE: OwnerCategory.OTHER_ENTERPRISE,
    VmiOwnerCategory.METSAHALLITUS: OwnerCategory.METSAHALLITUS,
    VmiOwnerCategory.OTHER_STATE_AGENCY: OwnerCategory.OTHER_STATE_AGENCY,
    VmiOwnerCategory.FOREST_COOP: OwnerCategory.FOREST_COOP,
    VmiOwnerCategory.MUNICIPALITY: OwnerCategory.MUNICIPALITY,
    VmiOwnerCategory.CONGREGATION: OwnerCategory.CONGREGATION,
    VmiOwnerCategory.OTHER_COMMUNITY: OwnerCategory.OTHER_COMMUNITY,
    VmiOwnerCategory.UNDIVIDED: OwnerCategory.UNDIVIDED
}


_soil_peatland_map = {
    VmiSoilPeatlandCategory.MINERAL_SOIL: SoilPeatlandCategory.MINERAL_SOIL,
    VmiSoilPeatlandCategory.SPRUCE_MIRE: SoilPeatlandCategory.SPRUCE_MIRE,
    VmiSoilPeatlandCategory.PINE_MIRE: SoilPeatlandCategory.PINE_MIRE,
    VmiSoilPeatlandCategory.TREELESS_MIRE: SoilPeatlandCategory.TREELESS_MIRE,
}


_site_type_map = {
    VmiSiteType.LEHTO: SiteType.VERY_RICH_SITE,
    VmiSiteType.LEHTOMAINEN_KANGAS: SiteType.RICH_SITE,
    VmiSiteType.TUOREKANGAS: SiteType.DAMP_SITE,
    VmiSiteType.KUIVAHKOKANGAS: SiteType.SUB_DRY_SITE,
    VmiSiteType.KUIVAKANGAS: SiteType.DRY_SITE,
    VmiSiteType.KARUKKOKANGAS: SiteType.BARREN_SITE,
    VmiSiteType.KALLIOMAA_TAI_HIETIKKO: SiteType.ROCKY_OR_SANDY_AREA,
    VmiSiteType.LAKIMETSA_TAI_TUNTURIHAVUMETSA: SiteType.LAKIMETSA_TAI_TUNTURIHAVUMETSA,
    VmiSiteType.TUNTURIKOIVIKKO: SiteType.TUNTURIKOIVIKKO,
    VmiSiteType.AVOTUNTURI: SiteType.OPEN_MOUNTAINS
}


_drainage_category_map = {
    VmiDrainageCategory.OJITTAMATON_KANGAS_TAI_SUO: DrainageCategory.UNDRAINED_MINERAL_SOIL_OR_MIRE,
    VmiDrainageCategory.OJITETTU_KANGAS: DrainageCategory.DITCHED_MINERAL_SOIL,
    VmiDrainageCategory.OJIKKO: DrainageCategory.DITCHED_MIRE,
    VmiDrainageCategory.MUUTTUMA: DrainageCategory.TRANSFORMING_MIRE,
    VmiDrainageCategory.TURVEKANGAS: DrainageCategory.TRANSFORMED_MIRE
}

_stratum_rank_map = {
    VmiStratumRank.UNGROWABLE_SAPLINGS: Storey.DOMINANT,
    VmiStratumRank.DOMINANT: Storey.DOMINANT,
    VmiStratumRank.OVER_1: Storey.OVER,
    VmiStratumRank.OVER_2: Storey.SPARE,
    VmiStratumRank.OVER_3: Storey.OVER,
    VmiStratumRank.UNDER_1: Storey.UNDER,
    VmiStratumRank.UNDER_2: Storey.UNDER,
    VmiStratumRank.UNDER_3: Storey.UNDER,
    VmiStratumRank.UNDER_4: Storey.UNDER,
    VmiStratumRank.REMOVAL: Storey.REMOVAL
}


_tree_storey_map = {
    VmiTreeStorey.DOMINANT_MAIN: Storey.DOMINANT,
    VmiTreeStorey.DOMINANT_MIDDLE: Storey.DOMINANT,
    VmiTreeStorey.DOMINANT_LOWER: Storey.DOMINANT,
    VmiTreeStorey.UNDER: Storey.UNDER,
    VmiTreeStorey.OVER_MAIN: Storey.OVER,
    VmiTreeStorey.OVER_OTHER: Storey.OVER,
    VmiTreeStorey.DOMINANT_SPARE_1: Storey.INDETERMINATE,
    VmiTreeStorey.DOMINANT_SPARE_2: Storey.INDETERMINATE,
    VmiTreeStorey.DOMINANT_SPARE_3: Storey.INDETERMINATE,
    VmiTreeStorey.UNDER_SPARE_1: Storey.INDETERMINATE,
    VmiTreeStorey.OVER_SPARE_1: Storey.SPARE,
    VmiTreeStorey.OVER_SPARE_2: Storey.SPARE
}

_origin_map = {
    VmiOrigin.UNKNOWN: Origin.UNKNOWN,
    VmiOrigin.NATURAL_SEED: Origin.NATURAL_SEED,
    VmiOrigin.NATURAL_SPROUT: Origin.NATURAL_SPROUT,
    VmiOrigin.PLANTED: Origin.PLANTED,
    VmiOrigin.SEEDED: Origin.SEEDED
}


_peatland_forest_type_map = {
    VmiPeatlandForestType.LEHTOKORPI: PeatlandForestType.LEHTOKORPI,
    VmiPeatlandForestType.RUOHOKORPI: PeatlandForestType.RUOHOKORPI,
    VmiPeatlandForestType.KANGASKORPI: PeatlandForestType.KANGASKORPI,
    VmiPeatlandForestType.MUSTIKKAKORPI: PeatlandForestType.MUSTIKKAKORPI,
    VmiPeatlandForestType.PUOLUKKAKORPI: PeatlandForestType.PUOLUKKAKORPI,
    VmiPeatlandForestType.PALLOSARAKORPI: PeatlandForestType.PALLOSARAKORPI,
    VmiPeatlandForestType.KORPIRAME: PeatlandForestType.KORPIRAME,
    VmiPeatlandForestType.PALLOSARARAME: PeatlandForestType.PALLOSARARAME,
    VmiPeatlandForestType.KANGASRAME: PeatlandForestType.KANGASRAME,
    VmiPeatlandForestType.ISOVARPURAME: PeatlandForestType.ISOVARPURAME,
    VmiPeatlandForestType.RAHKARAME: PeatlandForestType.RAHKARAME,
    VmiPeatlandForestType.VARSINAINEN_LEHTOKORPI: PeatlandForestType.VARSINAINEN_LEHTOKORPI,
    VmiPeatlandForestType.KOIVULETTOKORPI: PeatlandForestType.KOIVULETTOKORPI,
    VmiPeatlandForestType.RUOHOINEN_SARAKORPI: PeatlandForestType.RUOHOINEN_SARAKORPI,
    VmiPeatlandForestType.VARSINAINEN_SARAKORPI: PeatlandForestType.VARSINAINEN_SARAKORPI,
    VmiPeatlandForestType.VARSINAINEN_LEHTORAME: PeatlandForestType.VARSINAINEN_LEHTORAME,
    VmiPeatlandForestType.RUOHOINEN_SARARAME: PeatlandForestType.RUOHOINEN_SARARAME,
    VmiPeatlandForestType.VARSINAINEN_SARARAME: PeatlandForestType.VARSINAINEN_SARARAME,
    VmiPeatlandForestType.TUPAVILLASARARAME: PeatlandForestType.TUPAVILLASARARAME,
    VmiPeatlandForestType.LYHYTKORSIRAME: PeatlandForestType.LYHYTKORSIRAME,
    VmiPeatlandForestType.TUPAVILLARAME: PeatlandForestType.TUPAVILLARAME,
    VmiPeatlandForestType.KEIDASRAME: PeatlandForestType.KEIDASRAME,
    VmiPeatlandForestType.VARSINAINENLETTO: PeatlandForestType.VARSINAINENLETTO,
    VmiPeatlandForestType.RIMPILETTO: PeatlandForestType.RIMPILETTO,
    VmiPeatlandForestType.RUOHOINENSARANEVA: PeatlandForestType.RUOHOINENSARANEVA,
    VmiPeatlandForestType.RUOHOINENRIMPINEVA: PeatlandForestType.RUOHOINENRIMPINEVA,
    VmiPeatlandForestType.VARSINAINENSARANEVA: PeatlandForestType.VARSINAINENSARANEVA,
    VmiPeatlandForestType.VARSINAINENRIMPINEVA: PeatlandForestType.VARSINAINENRIMPINEVA,
    VmiPeatlandForestType.LYHYTKORSIKALVAKKANEVA: PeatlandForestType.LYHYTKORSIKALVAKKANEVA,
    VmiPeatlandForestType.LYHYTKORSINEVA: PeatlandForestType.LYHYTKORSINEVA,
    VmiPeatlandForestType.RAHKANEVA: PeatlandForestType.RAHKANEVA,
}

_drained_peatland_forest_type_map = {
    VmiDrainedPeatlandForestType.HERB_RICH_TYPE: DrainedPeatlandForestType.HERB_RICH_TYPE,
    VmiDrainedPeatlandForestType.VACCINIUM_MYRTILLUS_TYPE_1: DrainedPeatlandForestType.VACCINIUM_MYRTILLUS_TYPE_1,
    VmiDrainedPeatlandForestType.VACCINIUM_MYRTILLUS_TYPE_2: DrainedPeatlandForestType.VACCINIUM_MYRTILLUS_TYPE_2,
    VmiDrainedPeatlandForestType.VACCINIUM_VITIS_IDAEA_TYPE: DrainedPeatlandForestType.VACCINIUM_VITIS_IDAEA_TYPE,
    VmiDrainedPeatlandForestType.DEV_FROM_GENUINE_FORESTED_MIRE:
    DrainedPeatlandForestType.DEV_FROM_GENUINE_FORESTED_MIRE,
    VmiDrainedPeatlandForestType.DWARF_SHRUB_TYPE: DrainedPeatlandForestType.DWARF_SHRUB_TYPE,
    VmiDrainedPeatlandForestType.CLADONIA_TYPE: DrainedPeatlandForestType.CLADONIA_TYPE,
}


TREE_CATEGORY_MAP: dict[VmiTreeCategory, TreeCategory] = {
    VmiTreeCategory.C0: TreeCategory.C0,
    VmiTreeCategory.C1: TreeCategory.C1,
    VmiTreeCategory.C2: TreeCategory.C3,
    VmiTreeCategory.C3: TreeCategory.C3,
    VmiTreeCategory.C4: TreeCategory.C3,
    VmiTreeCategory.C5: TreeCategory.C7,
    VmiTreeCategory.C6: TreeCategory.C7,
    VmiTreeCategory.C7: TreeCategory.C7,
    VmiTreeCategory.C8: TreeCategory.C7,
    VmiTreeCategory.C9: TreeCategory.C3,

    VmiTreeCategory.A: TreeCategory.A,
    VmiTreeCategory.B: TreeCategory.B,
    VmiTreeCategory.C: TreeCategory.A,
    VmiTreeCategory.D: TreeCategory.D,
    VmiTreeCategory.E: TreeCategory.E,
    VmiTreeCategory.F: TreeCategory.F,
    VmiTreeCategory.G: TreeCategory.G,
}


def map_vmi_tree_category(raw: Optional[str]) -> Optional[TreeCategory]:
    if raw is None:
        return None

    raw = raw.strip()
    if not raw or raw == ".":
        return None

    try:
        vmi_enum = VmiTreeCategory(raw.upper())

    except ValueError as exc:
        raise ValueError(f'Unknown VMI tree_category: {raw}') from exc

    return TREE_CATEGORY_MAP.get(vmi_enum)


def is_empty_vmi_str(candidate: str) -> bool:
    return candidate in ('', ' ', '  ', '.', '\n')


def convert_drainage_category(code):
    if is_empty_vmi_str(code):
        return None
    value = VmiDrainageCategory(code)
    return _drainage_category_map.get(value)


def convert_site_type_category(code: str) -> Optional[SiteType]:
    if is_empty_vmi_str(code):
        return None
    value = VmiSiteType(code)
    return _site_type_map.get(value)


def convert_soil_peatland_category(code: str) -> Optional[SoilPeatlandCategory]:
    if is_empty_vmi_str(code):
        return None

    if code == '0':
        return None

    vmi_category = VmiSoilPeatlandCategory(code)
    return _soil_peatland_map.get(vmi_category)


def convert_land_use_category(lu_code: str) -> LandUseCategory:
    """sanitization of lu_code is the responsibility of the caller,
    meaning that this conversion will fail e.g. if the parameter is a lower-case letter."""
    vmi_category = VmiLandUseCategory(lu_code)
    return _land_use_map[vmi_category]


def convert_species(species_code: str) -> TreeSpecies:
    """Converts VMI species code to internal TreeSpecies code"""
    value = species_code.strip()
    vmi_species = VmiSpecies(value)
    return _species_map[vmi_species]


def convert_owner(owner_code: str) -> OwnerCategory:
    vmi_owner = VmiOwnerCategory(owner_code)
    return _owner_map[vmi_owner]


def convert_stratum_rank(rank_code: str) -> Optional[Storey]:
    if is_empty_vmi_str(rank_code):
        return None
    vmi_rank = VmiStratumRank(rank_code)
    return _stratum_rank_map[vmi_rank]


def convert_tree_storey(storey_code: str) -> Optional[Storey]:
    if is_empty_vmi_str(storey_code):
        return None
    vmi_storey = VmiTreeStorey(storey_code)
    return _tree_storey_map[vmi_storey]


def convert_origin(origin_code: str) -> Optional[Origin]:
    if is_empty_vmi_str(origin_code):
        return None
    vmi_origin = VmiOrigin(origin_code)
    return _origin_map.get(vmi_origin)


def convert_peatland_forest_type(code: str) -> Optional[PeatlandForestType]:

    if is_empty_vmi_str(code):
        return None

    if code == '0':
        return None

    vmi_code = VmiPeatlandForestType(code)
    return _peatland_forest_type_map[vmi_code]


def convert_drained_peatland_forest_type(code: str) -> Optional[DrainedPeatlandForestType]:
    if is_empty_vmi_str(code):
        return None

    if code == '0':
        return None
    vmi_code = VmiDrainedPeatlandForestType(code)
    return _drained_peatland_forest_type_map[vmi_code]
