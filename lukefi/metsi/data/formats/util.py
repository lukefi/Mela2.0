from enum import IntEnum
from typing import Optional, Any, Union
from lukefi.metsi.data.enums.internal import (
    DrainageCategory, LandUseCategory, OwnerCategory, SiteType, SoilPeatlandCategory, TreeSpecies)


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


def convert_str_to_type[T:(int, float, str, OwnerCategory, LandUseCategory, SoilPeatlandCategory, SiteType,
                           DrainageCategory, TreeSpecies)](value: str, ret_type: type[T]) -> Optional[T]:
    if value == "None":
        return None
    if issubclass(ret_type, IntEnum):
        return ret_type(int(value))
    return ret_type(value)


def basal_from_ppa(data_row: str, indices) -> float:
    # Basal area from ppa1..ppa5 (0.1 m2 units)
    ppas = []
    for k in ("ppa1", "ppa2", "ppa3", "ppa4", "ppa5"):

        ppas.append(get_or_default(parse_type(data_row[indices[k]], float), 0.0))

    return sum(ppas) / 10.0
