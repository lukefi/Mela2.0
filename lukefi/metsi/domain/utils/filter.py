from typing import Callable
import numpy as np
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

    if object_ == "stands":
        if verb == "select":
            return [s for s in stands if predicate(s)]
        # remove
        return [s for s in stands if not predicate(s)]

    if object_ == "trees":
        for s in stands:
            trees = s.reference_trees
            if trees.size == 0:
                continue

            mask = np.asarray(predicate(trees), dtype=bool)
            if mask.shape != (trees.size,):
                raise ValueError(
                    f"tree predicate must return mask of shape ({trees.size},), got {mask.shape}"
                )

            if verb == "remove":
                mask = ~mask

            s.reference_trees = trees[mask]
        return stands

    # handle strata
    for s in stands:
        strata = s.tree_strata
        if strata.size == 0:
            continue

        mask = np.asarray(predicate(strata), dtype=bool)
        if mask.shape != (strata.size,):
            raise ValueError(
                f"strata predicate must return mask of shape ({strata.size},), got {mask.shape}"
            )

        if verb == "remove":
            mask = ~mask

        s.tree_strata = strata[mask]
    return stands
