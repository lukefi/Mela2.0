from enum import Enum, IntEnum
from typing import Any


def parse_type[T:int | float | str](source, *ts: type[T]):
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


def parse_int(source: str | None) -> int | None:
    if source is None:
        return None
    try:
        return int(source)
    except (ValueError, TypeError):
        return None


def parse_float(source: str | None) -> float | None:
    if source is None:
        return None
    try:
        return float(source)
    except (ValueError, TypeError):
        return None


def get_or_default(maybe: Any | None, default: Any = None) -> Any:
    return default if maybe is None else maybe


def convert_str_to_type[T: Enum | int | float | str](value: str | None,
                                                          ret_type: type[T]) -> T | None:
    if value is None or value == "None":
        return None
    if issubclass(ret_type, IntEnum):
        return ret_type(int(value))
    if issubclass(ret_type, Enum):
        return ret_type(value)
    if issubclass(ret_type, int):
        return ret_type(value)
    if issubclass(ret_type, float):
        return ret_type(value)
    return ret_type(value)
