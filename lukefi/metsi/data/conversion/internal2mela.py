from copy import copy
from enum import IntEnum
from typing import Mapping

import numpy as np
import numpy.typing as npt

from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.enums.mela import (
    MelaOwnerCategory,
    MelaSiteTypeCategory,
    MelaSoilAndPeatlandCategory,
    MelaTreeOrigin,
    MelaTreeSpecies,
    MelaLandUseCategory,
    MelaDrainageCategory
)
from lukefi.metsi.data.enums.internal import (
    Origin,
    SiteType,
    SoilPeatlandCategory,
    TreeSpecies,
    OwnerCategory,
    LandUseCategory,
    DrainageCategory

)
from lukefi.metsi.data.conversion.util import apply_mappers
from lukefi.metsi.app.utils import MetsiException


_SPECIES_MAP = {
    TreeSpecies.PINE: MelaTreeSpecies.SCOTS_PINE,
    TreeSpecies.SPRUCE: MelaTreeSpecies.NORWAY_SPRUCE,
    TreeSpecies.SILVER_BIRCH: MelaTreeSpecies.SILVER_BIRCH,
    TreeSpecies.DOWNY_BIRCH: MelaTreeSpecies.DOWNY_BIRCH,
    TreeSpecies.ASPEN: MelaTreeSpecies.ASPEN,
    TreeSpecies.GREY_ALDER: MelaTreeSpecies.ALDER,
    TreeSpecies.COMMON_ALDER: MelaTreeSpecies.ALDER,
    TreeSpecies.OTHER_CONIFEROUS: MelaTreeSpecies.OTHER_CONIFEROUS,
    TreeSpecies.OTHER_DECIDUOUS: MelaTreeSpecies.OTHER_DECIDUOUS,
    TreeSpecies.DOUGLAS_FIR: MelaTreeSpecies.OTHER_CONIFEROUS,
    TreeSpecies.JUNIPER: MelaTreeSpecies.OTHER_CONIFEROUS,
    TreeSpecies.SHORE_PINE: MelaTreeSpecies.OTHER_CONIFEROUS,
    TreeSpecies.EUROPEAN_WHITE_ELM: MelaTreeSpecies.OTHER_DECIDUOUS,
    TreeSpecies.LARCH: MelaTreeSpecies.OTHER_CONIFEROUS,
    TreeSpecies.SMALL_LEAVED_LIME: MelaTreeSpecies.OTHER_DECIDUOUS,
    TreeSpecies.BLACK_SPRUCE: MelaTreeSpecies.OTHER_CONIFEROUS,
    TreeSpecies.WILLOW: MelaTreeSpecies.OTHER_DECIDUOUS,
    TreeSpecies.MOUNTAIN_ASH: MelaTreeSpecies.OTHER_DECIDUOUS,
    TreeSpecies.ABIES: MelaTreeSpecies.OTHER_CONIFEROUS,
    TreeSpecies.GOAT_WILLOW: MelaTreeSpecies.OTHER_DECIDUOUS,
    TreeSpecies.COMMON_ASH: MelaTreeSpecies.OTHER_DECIDUOUS,
    TreeSpecies.KEDAR: MelaTreeSpecies.OTHER_CONIFEROUS,
    TreeSpecies.SERBIAN_SPRUCE: MelaTreeSpecies.OTHER_CONIFEROUS,
    TreeSpecies.OAK: MelaTreeSpecies.OTHER_DECIDUOUS,
    TreeSpecies.BIRD_CHERRY: MelaTreeSpecies.OTHER_DECIDUOUS,
    TreeSpecies.MAPLE: MelaTreeSpecies.OTHER_DECIDUOUS,
    TreeSpecies.CURLY_BIRCH: MelaTreeSpecies.OTHER_DECIDUOUS,
    TreeSpecies.WYCH_ELM: MelaTreeSpecies.OTHER_DECIDUOUS,
    TreeSpecies.UNKNOWN_CONIFEROUS: MelaTreeSpecies.OTHER_CONIFEROUS,
    TreeSpecies.UNKNOWN_DECIDUOUS: MelaTreeSpecies.OTHER_DECIDUOUS,
    TreeSpecies.OTHER_PINE: MelaTreeSpecies.OTHER_CONIFEROUS,
    TreeSpecies.OTHER_SPRUCE: MelaTreeSpecies.OTHER_CONIFEROUS,
    TreeSpecies.THUJA: MelaTreeSpecies.OTHER_CONIFEROUS,
    TreeSpecies.YEW: MelaTreeSpecies.OTHER_CONIFEROUS,
    TreeSpecies.BAY_WILLOW: MelaTreeSpecies.OTHER_DECIDUOUS,
    TreeSpecies.POPLAR: MelaTreeSpecies.OTHER_DECIDUOUS,
    TreeSpecies.HAZEL: MelaTreeSpecies.OTHER_DECIDUOUS
}

_ORIGIN_MAP = {
    Origin.UNSET: MelaTreeOrigin.NATURAL,
    Origin.NATURAL: MelaTreeOrigin.NATURAL,
    Origin.SEEDED: MelaTreeOrigin.SEEDED,
    Origin.PLANTED: MelaTreeOrigin.PLANTED
}

_LAND_USE_MAP = {
    LandUseCategory.FOREST: MelaLandUseCategory.FOREST_LAND,
    LandUseCategory.SCRUB_LAND: MelaLandUseCategory.SCRUB_LAND,
    LandUseCategory.WASTE_LAND: MelaLandUseCategory.WASTE_LAND,
    LandUseCategory.OTHER_FOREST: MelaLandUseCategory.OTHER,
    LandUseCategory.AGRICULTURAL: MelaLandUseCategory.AGRICULTURAL_LAND,
    LandUseCategory.BUILT_LAND: MelaLandUseCategory.BUILT_UP_LAND,
    LandUseCategory.ROAD: MelaLandUseCategory.ROADS_OR_ELECTRIC_LINES,
    LandUseCategory.ENERGY_TRANSMISSION_LINE: MelaLandUseCategory.ROADS_OR_ELECTRIC_LINES,
    LandUseCategory.FRESHWATER: MelaLandUseCategory.LAKES_AND_RIVERS,
    LandUseCategory.SEA: MelaLandUseCategory.SEA,
    LandUseCategory.REAL_ESTATE: MelaLandUseCategory.BUILT_UP_LAND,
    LandUseCategory.OTHER_LAND: MelaLandUseCategory.ROADS_OR_ELECTRIC_LINES,
    LandUseCategory.WATER_BODY: MelaLandUseCategory.LAKES_AND_RIVERS
}

_OWNER_MAP = {
    OwnerCategory.UNKNOWN: MelaOwnerCategory.PRIVATE,
    OwnerCategory.PRIVATE: MelaOwnerCategory.PRIVATE,
    OwnerCategory.FOREST_INDUSTRY: MelaOwnerCategory.ENTERPRISE,
    OwnerCategory.OTHER_ENTERPRISE: MelaOwnerCategory.ENTERPRISE,
    OwnerCategory.METSAHALLITUS: MelaOwnerCategory.STATE,
    OwnerCategory.OTHER_STATE_AGENCY: MelaOwnerCategory.STATE,
    OwnerCategory.FOREST_COOP: MelaOwnerCategory.COMMUNITY,
    OwnerCategory.MUNICIPALITY: MelaOwnerCategory.MUNICIPALITY,
    OwnerCategory.CONGREGATION: MelaOwnerCategory.COMMUNITY,
    OwnerCategory.OTHER_COMMUNITY: MelaOwnerCategory.COMMUNITY,
    OwnerCategory.UNDIVIDED: MelaOwnerCategory.COMMUNITY
}


_SITE_TYPE_MAP = {
    SiteType.VERY_RICH_SITE: MelaSiteTypeCategory.VERY_RICH_SITE,
    SiteType.RICH_SITE: MelaSiteTypeCategory.RICH_SITE,
    SiteType.DAMP_SITE: MelaSiteTypeCategory.DAMP_SITE,
    SiteType.SUB_DRY_SITE: MelaSiteTypeCategory.SUB_DRY_SITE,
    SiteType.DRY_SITE: MelaSiteTypeCategory.DRY_SITE,
    SiteType.BARREN_SITE: MelaSiteTypeCategory.BARREN_SITE,
    SiteType.ROCKY_OR_SANDY_AREA: MelaSiteTypeCategory.ROCKY_OR_SANDY_AREA,
    SiteType.OPEN_MOUNTAINS: MelaSiteTypeCategory.OPEN_MOUNTAINS,
    SiteType.TUNTURIKOIVIKKO: MelaSiteTypeCategory.OPEN_MOUNTAINS,
    SiteType.LAKIMETSA_TAI_TUNTURIHAVUMETSA: MelaSiteTypeCategory.OPEN_MOUNTAINS
}

# this doesn't have a mapping for TREELESS_MIRE, as its mapping to MELA values is determined by the SiteType category.
_SOIL_PEATLAND_MAP = {
    SoilPeatlandCategory.MINERAL_SOIL: MelaSoilAndPeatlandCategory.MINERAL_SOIL,
    SoilPeatlandCategory.SPRUCE_MIRE: MelaSoilAndPeatlandCategory.PEATLAND_SPRUCE_MIRE,
    SoilPeatlandCategory.PINE_MIRE: MelaSoilAndPeatlandCategory.PEATLAND_PINE_MIRE,
}

_RICH_MIRE_TYPES = [
    SiteType.VERY_RICH_SITE,
    SiteType.RICH_SITE,
    SiteType.DAMP_SITE
]


def site_type_mapper(target):
    target.site_type_category = _SITE_TYPE_MAP.get(target.site_type_category)
    return target


def drainage_category_mapper(target):
    if target.drainage_category == DrainageCategory.UNDRAINED_MINERAL_SOIL_OR_MIRE:
        if target.soil_peatland_category == SoilPeatlandCategory.MINERAL_SOIL:
            target.drainage_category = MelaDrainageCategory.UNDRAINED_MINERAL_SOIL
        else:
            target.drainage_category = MelaDrainageCategory.UNDRAINED_MIRE
    elif target.drainage_category == DrainageCategory.DITCHED_MINERAL_SOIL:
        target.drainage_category = MelaDrainageCategory.DITCHED_MINERAL_SOIL
    elif target.drainage_category == DrainageCategory.DITCHED_MIRE:
        target.drainage_category = MelaDrainageCategory.DITCHED_MIRE
    elif target.drainage_category == DrainageCategory.TRANSFORMING_MIRE:
        target.drainage_category = MelaDrainageCategory.TRANSFORMING_MIRE
    elif target.drainage_category == DrainageCategory.TRANSFORMED_MIRE:
        target.drainage_category = MelaDrainageCategory.TRANSFORMED_MIRE
    else:
        target.drainage_category = MelaDrainageCategory.UNDRAINED_MINERAL_SOIL
    return target


def soil_peatland_mapper(target):
    """
    If the internal SoilPeatlandCategory is TREELESS_MIRE, determining the soil or peatland type for MELA
    requires knowing the site type (fertility type). Make sure to set it first, because otherwise this method
    is unable to determine soil_peatland_category and sets it to None.
    """

    if target.soil_peatland_category == SoilPeatlandCategory.TREELESS_MIRE:
        if target.site_type_category is None:
            target.soil_peatland_category = None

        elif target.site_type_category in _RICH_MIRE_TYPES:
            target.soil_peatland_category = MelaSoilAndPeatlandCategory.PEATLAND_RICH_TREELESS_MIRE
        else:
            target.soil_peatland_category = MelaSoilAndPeatlandCategory.PEATLAND_BARREN_TREELESS_MIRE
    else:
        target.soil_peatland_category = _SOIL_PEATLAND_MAP.get(target.soil_peatland_category)

    return target


def land_use_mapper(target):
    """in-place mapping from internal LandUseCategory to MelaLandUseCategory"""
    target.land_use_category = _LAND_USE_MAP.get(target.land_use_category)
    return target


def owner_mapper(target):
    """in-place mapping from internal land owner category to mela owner category"""
    target.owner_category = _OWNER_MAP.get(target.owner_category)
    return target


def mela_trees(trees: ReferenceTrees) -> ReferenceTrees:
    """
    Convert SoA ReferenceTrees so that enumerated category variables are converted to Mela value space.
    """
    rt = copy(trees)

    rt.species = _convert_vector(rt.species, _SPECIES_MAP)
    rt.origin = _convert_vector(rt.origin, _ORIGIN_MAP)

    return rt


def _convert_vector[T: IntEnum](array: npt.NDArray[np.int32],
                                conversions: Mapping[T, IntEnum]) -> npt.NDArray[np.int32]:
    for i, original_value in enumerate(array):
        array[i] = conversions[original_value]
    return array


def stand_location_converter(target):
    """
    in-place conversion of ForestStand geolocation to kilometer precision,
    and to YKJ/KKJ3 with band prefix 3 removed for EPSG:2393
    """
    if target.geo_location[3] == 'EPSG:3067':
        lat, lon = (target.geo_location[0] / 1000, target.geo_location[1] / 1000)
    elif target.geo_location[3] == 'EPSG:2393':
        lat, lon = (target.geo_location[0] / 1000, target.geo_location[1] / 1000 - 3000)
    else:
        raise MetsiException(f"Unsupported CRS {target.geo_location[3]} for stand {target.identifier}")

    target.geo_location = (
        lat,
        lon,
        target.geo_location[2],
        target.geo_location[3])
    return target


def mela_stand(stand: ForestStand) -> ForestStand:
    """Convert a ForestStand so that enumerated category variables are converted to Mela value space"""
    result = copy(stand)
    result.geo_location = copy(stand.geo_location)
    result.area_weight_factors = copy(stand.area_weight_factors)
    result = apply_mappers(result, *default_mela_stand_mappers)

    # Some  fixed RST spesific classifier conversions TODO: find a better place for these.
    # stand level
    result.forestry_centre_id = (-1 if result.forestry_centre_id is None
                                 else result.forestry_centre_id)
    result.municipality_id = (-1 if result.municipality_id is None
                              else result.municipality_id)

    result.reference_trees = mela_trees(result.reference_trees)

    return result


default_mela_stand_mappers = [stand_location_converter,
                              owner_mapper,
                              land_use_mapper,
                              site_type_mapper,
                              soil_peatland_mapper,
                              drainage_category_mapper]
