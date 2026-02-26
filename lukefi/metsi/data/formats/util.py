from enum import IntEnum
from typing import Optional, Any, Union
from lukefi.metsi.data.enums.internal import (
    ArableLandDetail,
    BuildUpLandDetail,
    DrainageCategory,
    FraLandUseClass,
    InlandWaterDetail,
    LandUseCategory,
    LandUseCategoryDetail,
    OtherForestryLandDetail,
    OwnerCategory,
    PoorlyProductiveForestLandDetail,
    ProductiveForestLandDetail,
    SiteType,
    SoilPeatlandCategory,
    TreeSpecies,
    UnproductiveLandDetail)


def parse_type[T:Union[int, float, str]](source, *ts: type[T]):
    ''' Generic version of  parse_int and parse_float utilities'''
    ts_ = list(ts)
    try:
        t0 = ts_.pop(0)
        r = t0(source)
        for t in ts_:
            r = t(r)
        return r
    except (ValueError, TypeError, IndexError):
        return None


def parse_int(source: str | None) -> Optional[int]:
    if source is None:
        return None
    try:
        return int(source)
    except (ValueError, TypeError):
        return None


def parse_float(source: str | None) -> Optional[float]:
    if source is None:
        return None
    try:
        return float(source)
    except (ValueError, TypeError):
        return None


def get_or_default(maybe: Optional[Any], default: Any = None) -> Any:
    return default if maybe is None else maybe


def convert_str_to_type[T:(int,
                           float,
                           str,
                           OwnerCategory,
                           LandUseCategory,
                           SoilPeatlandCategory,
                           SiteType,
                           DrainageCategory,
                           TreeSpecies,
                           FraLandUseClass)](value: str,
                                             ret_type: type[T]) -> Optional[T]:
    if value == "None":
        return None
    if issubclass(ret_type, IntEnum):
        return ret_type(int(value))
    return ret_type(value)


def convert_land_use_category_detail(
        land_use_category: Optional[LandUseCategory],
        code: str) -> Optional[LandUseCategoryDetail]:
    if land_use_category is None or code == 'None':
        return None
    if land_use_category == LandUseCategory.FOREST:
        return ProductiveForestLandDetail(int(code))
    if land_use_category == LandUseCategory.SCRUB_LAND:
        return PoorlyProductiveForestLandDetail(int(code))
    if land_use_category == LandUseCategory.WASTE_LAND:
        return UnproductiveLandDetail(int(code))
    if land_use_category == LandUseCategory.OTHER_FOREST:
        return OtherForestryLandDetail(int(code))
    if land_use_category == LandUseCategory.AGRICULTURAL:
        return ArableLandDetail(int(code))
    if land_use_category == LandUseCategory.BUILT_LAND:
        return BuildUpLandDetail(int(code))
    if land_use_category == LandUseCategory.FRESHWATER:
        return InlandWaterDetail(int(code))
    return None
