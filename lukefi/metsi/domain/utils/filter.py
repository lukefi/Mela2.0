from typing import Callable
from lukefi.metsi.domain.forestry_types import StandList

VERBS: set[str] = {"select", "remove"}
OBJECTS: set[str] = {"stands", "trees", "strata"}


def parsecommand(command: str) -> tuple[str, str]:
    parts = command.split()
    if len(parts) > 2:
        raise ValueError(f"filter syntax error: {command}")
    if len(parts) == 1:
        v, o = parts[0], "stands"
    else:
        v, o = parts
    if v not in VERBS:
        raise ValueError(f"invalid filter verb: {v} (in filter {command})")
    if o not in OBJECTS:
        raise ValueError(f"invalid filter object: {o} (in filter {command})")
    return v, o


def applyfilter(stands: StandList, command: str, predicate: Callable[..., bool]) -> StandList:
    verb, object_ = parsecommand(command)
    if verb == "remove":
        p = predicate
        predicate = lambda f: not p(f)  # pylint: disable=unnecessary-lambda-assignment
    if object_ == "stands":
        stands = [
            s
            for s in stands
            if predicate(s)
        ]
    elif object_ == "trees":
        for s in stands:
            s.reference_trees_pre_vec = [
                t
                for t in s.reference_trees_pre_vec
                if predicate(t)
            ]
    elif object_ == "strata":
        for s in stands:
            s.tree_strata_pre_vec = [
                t
                for t in s.tree_strata_pre_vec
                if predicate(t)
            ]
    return stands
