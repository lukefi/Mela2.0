from typing import Any
import re
import numpy as np
import numpy.typing as npt
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.data.enums.internal import (
    TreeSpecies,
    Origin,
    Storey
)


UT_SPECIES_FIELDS = [
    ("ma", TreeSpecies.PINE),
    ("ku", TreeSpecies.SPRUCE),
    ("ra", TreeSpecies.SILVER_BIRCH),
    ("hi", TreeSpecies.DOWNY_BIRCH),
    ("ha", TreeSpecies.ASPEN),
    ("hl", TreeSpecies.GREY_ALDER),
    ("tl", TreeSpecies.OTHER_DECIDUOUS),
    ("mh", TreeSpecies.OTHER_CONIFEROUS),
    ("ml", TreeSpecies.OTHER_DECIDUOUS),
    ("_10", TreeSpecies.UNSET),
]

UT_CATEGORIES = [
    ("kkp", "usable"),
    ("klv", "unusable"),
    ("vlj", "farmed"),
]


def find_sapling_reference_tree_index(rt: ReferenceTrees, osid: int) -> int | None:
    target_osid = int(osid)

    for i in reference_tree_indices_by_stratum(rt, osid):
        if not bool(rt.sapling[i]):
            continue
        try:
            if int(rt.stratum[i]) == target_osid:
                return i
        except (TypeError, ValueError):
            continue

    return None


def storey_from_layer(stand: ForestStand, layer: int) -> int:
    strata = getattr(stand, "tree_strata", None)
    if strata is None or layer >= strata.size:
        return int(Storey.UNSET)

    try:
        v = int(strata.storey[layer])
        return v if v >= 0 else int(Storey.UNSET)
    except (TypeError, ValueError):
        return int(Storey.UNSET)


def reference_tree_indices_by_stratum(rt: ReferenceTrees, osid: int) -> list[int]:
    target = str(int(osid))
    retval: list[int] = []
    for i, value in enumerate(rt.stratum.tolist()):
        if str(value) == target:
            retval.append(i)
    return retval


def reference_tree_index_by_osid(rt: ReferenceTrees, osid: int) -> int | None:
    """Backward-compatible helper: returns the first match for a stratum id."""
    matches = reference_tree_indices_by_stratum(rt, osid)
    return matches[0] if matches else None


def find_reference_tree_index_by_tree_number(rt: ReferenceTrees, tree_number: int) -> int | None:
    target = int(tree_number)
    for i, value in enumerate(rt.tree_number.tolist()):
        if bool(rt.sapling[i]):
            continue
        try:
            if int(value) == target:
                return i
        except (TypeError, ValueError):
            continue
    return None


def find_non_sapling_reference_tree_index(rt: ReferenceTrees, osid: int, tree_number: int) -> int | None:
    target_tree_number = int(tree_number)
    for i in reference_tree_indices_by_stratum(rt, osid):
        if bool(rt.sapling[i]):
            continue
        try:
            if int(rt.tree_number[i]) == target_tree_number:
                return i
        except (TypeError, ValueError):
            continue
    return None


def parse_int_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        x = int(float(value))
        return x if x > 0 else None
    except (TypeError, ValueError):
        return None


def next_osite_id(stand: ForestStand) -> int:
    used: list[int] = []

    rt = stand.reference_trees
    if rt is not None and rt.size > 0:
        for v in rt.stratum.tolist():
            x = parse_int_id(v)
            if x is not None:
                used.append(x)

    ms = getattr(stand, "motti_state", None)
    if ms is not None and ms.buffers is not None:
        ut = ms.buffers.saplings
        for layer in range(10):
            for spe_name, _ in UT_SPECIES_FIELDS:
                s = getattr(ut[0][layer], spe_name)
                for cat_code, _ in UT_CATEGORIES:
                    x = parse_int_id(getattr(s, f"osid_{cat_code}", 0))
                    if x is not None:
                        used.append(x)

    if ms is not None and getattr(ms, "yp", None) is not None:
        for i in range(int(getattr(ms, "ntrees", 0) or 0)):
            x = parse_int_id(getattr(ms.yp[0][i], "sid", 0))
            if x is not None:
                used.append(x)

    print("Used SID:s:", used)
    return (max(used) + 1) if used else 1


def safe_origin(raw: float | int | None) -> int:
    if not raw:
        return int(Origin.UNSET)
    try:
        v = int(raw)
        return v if v >= 0 else int(Origin.UNSET)
    except (ValueError, TypeError):
        return int(Origin.UNSET)


def next_reference_tree_number(rt: ReferenceTrees) -> int:
    vals = []
    for v in rt.tree_number.tolist():
        try:
            iv = int(v)
            if iv > 0:
                vals.append(iv)
        except (TypeError, ValueError):
            pass
    return (max(vals) + 1) if vals else 1


def next_reference_tree_identifier_suffix(stand: ForestStand) -> int:
    rt = stand.reference_trees
    used = set()

    suffix_re = re.compile(rf"^{re.escape(stand.identifier)}-(\d+)-tree$")

    for ident in rt.identifier.tolist():
        s = str(ident)
        m = suffix_re.match(s)
        if m:
            try:
                used.add(int(m.group(1)))
            except ValueError:
                pass

    n = 1
    while n in used:
        n += 1
    return n


def new_reference_tree_identity(stand: ForestStand) -> tuple[str, int]:
    rt = stand.reference_trees

    # keep tree_number allocation logic for Motti bookkeeping
    tree_number = next_reference_tree_number(rt)

    # allocate identifier independently so it is always unique in the stand
    ident_suffix = next_reference_tree_identifier_suffix(stand)
    identifier = f"{stand.identifier}-{ident_suffix}-tree"

    return identifier, tree_number


def update_stand_growth(stand: ForestStand,
                        diameters: npt.NDArray[np.float64],
                        heights: npt.NDArray[np.float64],
                        stems: npt.NDArray[np.float64],
                        step: int,
                        update_sapling: bool = True):
    """In-place update stand's reference trees with given diameters, heights and stem count.
    Increase ages for trees and stand. Remove sapling flag from trees that have grown beyond 1.3m. """
    if stand.reference_trees is None:
        raise MetsiException("Data not vectorized")

    trees = stand.reference_trees

    trees.biological_age = trees.biological_age + step
    trees.breast_height_age = np.where(
        (trees.height < 1.3) & (1.3 <= heights),
        trees.biological_age,
        trees.breast_height_age)
    trees.breast_height_diameter = diameters
    trees.height = heights
    trees.stems_per_ha = stems

    if update_sapling:
        trees.sapling = np.where(
            trees.height >= 1.3,
            False,
            trees.sapling)

    stand.year = (stand.year or 0) + step


def safe_storey_value(v: Any) -> float:
    if v is None:
        return 0.0
    return float(getattr(v, "value", v))
