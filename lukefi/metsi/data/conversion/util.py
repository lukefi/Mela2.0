from collections.abc import Callable
from typing import TypeVar
from lukefi.metsi.data.vector_model import VectorData

V = TypeVar("V", bound=VectorData)


def apply_mappers(target, *mappers: Callable):
    """apply a list of mapper functions to a target object"""
    for mapper in mappers:
        target = mapper(target)
    return target
