from collections.abc import Callable
from copy import copy
from typing import TypeVar
from lukefi.metsi.data.vector_model import VectorData

V = TypeVar("V", bound=VectorData)


def apply_mappers(target, *mappers: Callable):
    """apply a list of mapper functions to a target object"""
    for mapper in mappers:
        target = mapper(target)
    return target


def copy_vector_data(v: V) -> V:
    """Shallow-copy VectorData object + deep-copy its numpy arrays."""
    out = copy(v)
    for key in v.dtypes:
        arr = getattr(v, key, None)
        if arr is not None:
            setattr(out, key, arr.copy())
    out.size = v.size
    return out
