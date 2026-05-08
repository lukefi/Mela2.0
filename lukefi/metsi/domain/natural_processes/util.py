from typing import Any
import re
import numpy as np
import numpy.typing as npt
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

FDM_TO_MOTTI_STOREY = {
    Storey.DOMINANT: 2,  # ylempi
    Storey.UNDER: 1,     # alempi
    Storey.OVER: 3,      # siemenpuu
    Storey.SPARE: 4,     # säästöpuu
}

MOTTI_TO_FDM_STOREY = {
    1: int(Storey.UNDER),
    2: int(Storey.DOMINANT),
    3: int(Storey.OVER),
    4: int(Storey.SPARE),
}


def storey_from_motti(value: Any) -> int:
    try:
        return MOTTI_TO_FDM_STOREY.get(int(float(value)), int(Storey.UNSET))
    except (TypeError, ValueError):
        return int(Storey.UNSET)


def storey_to_motti(
    stand: ForestStand,
    index: int,
    fdm_storey: Storey,
    *,
    is_stratum_index: bool = False,
) -> int:
    """
    Convert FDM Storey -> Motti puustojakso/puuluokka.

    Exact classes:
      DOMINANT -> 2
      UNDER    -> 1
      OVER     -> 3
      SPARE    -> 4

    Fallback:
      - if only one stratum: ylempi=2
      - if multiple strata and this stratum is clearly lower:
          height gap > 5m and lower stratum height < 10m -> alempi=1
      - otherwise ylempi=2

    Parameters
    ----------
    index:
        If is_stratum_index=True, this is a direct tree_strata row index.
        Otherwise it is assumed to be a reference_trees row index, and the
        matching stratum row is resolved through rt.stratum -> strata.stratum_number.
    """
    if fdm_storey in FDM_TO_MOTTI_STOREY:
        return FDM_TO_MOTTI_STOREY[fdm_storey]

    strata = stand.tree_strata
    if strata is None or strata.size <= 1:
        return 2

    stratum_idx: int | None = None

    if is_stratum_index:
        if 0 <= index < strata.size:
            stratum_idx = index
    else:
        rt = stand.reference_trees
        if 0 <= index < rt.size:
            target_sid = int(rt.stratum[index])
            if target_sid is not None:
                for j in range(strata.size):
                    sid = int(strata.stratum_number[j])
                    if sid == target_sid:
                        stratum_idx = j
                        break

    if stratum_idx is None:
        return 2

    heights = np.nan_to_num(strata.mean_height, nan=0.0)
    current_h = float(heights[stratum_idx])
    max_h = float(np.max(heights))

    if (max_h - current_h) > 5.0 and current_h < 10.0:
        return 1

    return 2


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


def next_osite_id(stand: ForestStand) -> int:
    used: list[int] = []

    rt = stand.reference_trees
    if rt.size > 0:
        for v in rt.stratum:
            used.append(int(v))

    ms = stand.motti_state
    if ms is not None and ms.buffers is not None:
        ut = ms.buffers.saplings
        for layer in range(10):
            for spe_name, _ in UT_SPECIES_FIELDS:
                s = getattr(ut[0][layer], spe_name)
                for cat_code, _ in UT_CATEGORIES:
                    x = int(getattr(s, f"osid_{cat_code}", 0))
                    if x is not None:
                        used.append(x)

    if ms is not None and ms.yp is not None:
        for i in range(int(ms.ntrees or 0)):
            x = int(ms.yp[0][i].sid)
            if x is not None:
                used.append(x)

    return (max(used) + 1) if used else 1


def safe_origin(raw: float | int) -> int:
    v = int(raw)
    return v if v >= 0 else int(Origin.UNSET)

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
