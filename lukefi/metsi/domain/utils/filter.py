from enum import StrEnum
from typing import Callable
import numpy as np
import numpy.typing as npt
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.forestry_types import StandList


class Verb(StrEnum):
    SELECT = "select"
    REMOVE = "remove"


def filter_stands(stands: StandList, verb: str, predicate: Callable[[ForestStand], bool]) -> StandList:
    verb = Verb(verb)

    if verb == Verb.REMOVE:
        p = predicate
        predicate = lambda f: not p(f)  # pylint: disable=unnecessary-lambda-assignment

    stands = [s for s in stands if predicate(s)]
    return stands


def filter_trees(stands: StandList, mask: Callable[[ForestStand], npt.NDArray[np.bool_]]) -> StandList:
    for stand in stands:
        stand.reference_trees = stand.reference_trees[mask(stand)]
    return stands


def filter_strata(stands: StandList, mask: Callable[[ForestStand], npt.NDArray[np.bool_]]) -> StandList:
    for stand in stands:
        stand.tree_strata = stand.tree_strata[mask(stand)]
    return stands
