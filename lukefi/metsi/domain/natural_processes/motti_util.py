from typing import Any

import numpy as np
import numpy.typing as npt

from lukefi.metsi.data.enums.internal import Origin, Storey, TreeSpecies
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.motti.motti_types import Motti4SaplingStratum
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.domain.natural_processes.util import new_reference_tree_identity
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL


UT_CATEGORIES = [
    ("kkp", "usable"),
    ("klv", "unusable"),
    ("vlj", "farmed"),
]


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


def _safe_origin(raw: float | int) -> int:
    v = int(raw)
    return v if v >= 0 else int(Origin.UNSET)


def next_osite_id(stand: ForestStand) -> int:
    used: list[int] = []

    rt = stand.reference_trees
    if rt.size > 0:
        for v in rt.stratum:
            x = int(v)
            if x > 0:
                used.append(int(v))

    ms = stand.motti_state
    if ms is not None and ms.buffers is not None:
        ut = ms.buffers.saplings
        for layer in range(10):
            for spe_name, _ in UT_SPECIES_FIELDS:
                s = getattr(ut[0][layer], spe_name)
                for cat_code, _ in UT_CATEGORIES:
                    x = int(getattr(s, f"osid_{cat_code}", 0))
                    if x > 0:
                        used.append(x)

    if ms is not None and ms.yp is not None:
        for i in range(ms.ntrees or 0):
            x = int(ms.yp[0][i].sid)
            if x > 0:
                used.append(x)

    return (max(used) + 1) if used else 1


def _reference_tree_indices_by_stratum(rt: ReferenceTrees, osid: int) -> list[int]:
    target = str(osid)
    retval: list[int] = []
    for i, value in enumerate(rt.stratum.tolist()):
        if str(value) == target:
            retval.append(i)
    return retval


def _find_non_sapling_reference_tree_index(rt: ReferenceTrees, osid: int, tree_number: int) -> int | None:
    target_tree_number = tree_number
    for i in _reference_tree_indices_by_stratum(rt, osid):
        if bool(rt.sapling[i]):
            continue
        try:
            if int(rt.tree_number[i]) == target_tree_number:
                return i
        except (TypeError, ValueError):
            continue
    return None


def _storey_from_layer(stand: ForestStand, layer: int) -> int:
    # NOTE: This can't be right! Check.
    strata = getattr(stand, "tree_strata", None)
    if strata is None or layer >= strata.size:
        return int(Storey.UNSET)

    try:
        v = int(strata.storey[layer])
        return v if v >= 0 else int(Storey.UNSET)
    except (TypeError, ValueError):
        return int(Storey.UNSET)


def _find_sapling_reference_tree_index(rt: ReferenceTrees, osid: int) -> int | None:
    target_osid = osid

    for i in _reference_tree_indices_by_stratum(rt, osid):
        if not bool(rt.sapling[i]):
            continue
        try:
            if int(rt.stratum[i]) == target_osid:
                return i
        except (TypeError, ValueError):
            continue

    return None


def sync_yp_to_reference_trees(stand: ForestStand) -> None:
    ms = stand.motti_state
    if ms is None or ms.yp is None:
        return

    yp = ms.yp
    rt = stand.reference_trees

    for i in range(ms.ntrees):
        t = yp[0][i]

        sid = int(t.sid)
        if sid <= 0:
            continue
        yp_tree_id = int(t.id)

        if yp_tree_id <= 0:
            identifier, tree_number = new_reference_tree_identity(stand)
            yp_tree_id = tree_number
            t.id = float(tree_number)
            idx = None
            storey = int(Storey.UNSET)
        else:
            idx = _find_non_sapling_reference_tree_index(rt, sid, yp_tree_id)

            if idx is None:
                identifier, tree_number = new_reference_tree_identity(stand)
                yp_tree_id = tree_number
                t.id = float(tree_number)
                storey = int(Storey.UNSET)
            else:
                identifier = str(rt.identifier[idx])
                tree_number = int(rt.tree_number[idx])
                storey = int(rt.storey[idx]) if int(rt.storey[idx]) >= 0 else int(Storey.UNSET)

        row = {
            "identifier": identifier, # NOTE: should not be updated
            "tree_number": yp_tree_id, # NOTE: should not be updated
            "stratum": str(sid), # NOTE: should not be updated
            "species": int(t.spe), # NOTE: should not be updated
            "stems_per_ha": t.f,
            "origin": int(t.snt) - 1, # NOTE: should not be updated
            "height": t.h,
            "breast_height_diameter": t.d13,
            "biological_age": t.age,
            "breast_height_age": t.age13,
            "sapling": False, # Tarviiko tätä ollenkaan?
            "tree_category": "", # Tarviiko tätä olla tässä jos kerran tyhjä?
            "management_category": 1, # NOTE: should not be updated
            "storey": storey, # NOTE: should not be updated
            "basal_area": (t.ba / 10000.0) if getattr(t, "ba", None) is not None else 0.0,
            "volume": t.vol if getattr(t, "vol", None) is not None else 0.0,
        }

        if idx is None:
            rt.create(row)
        else:
            rt.update(row, idx)


def _build_reference_tree_update(*,
                                 identifier: str,
                                 tree_number: int,
                                 osid: int,
                                 species: TreeSpecies,
                                 category_code: str,
                                 stems_per_ha: float,
                                 origin_raw: float,
                                 height: float,
                                 diameter: float,
                                 age: float,
                                 age13: float,
                                 basal_area: float,
                                 volume: float,
                                 storey: int,
                                 ) -> dict[str, Any]:
    # NOTE: This could be raised into upper function stack
    return {
        "identifier": identifier,
        "tree_number": tree_number,
        "stratum": str(osid),
        "species": int(species),
        "stems_per_ha": stems_per_ha,
        "origin": _safe_origin(origin_raw),
        "height": height,
        "breast_height_diameter": diameter,
        "biological_age": age,
        "breast_height_age": age13,
        "sapling": True,
        "tree_category": category_code,
        "management_category": 1,
        "storey": storey,
        "basal_area": basal_area,
        "volume": volume,
    }


def sync_ut_to_reference_trees(stand: ForestStand) -> None:
    ms = stand.motti_state
    if ms is None or ms.buffers is None:
        return

    ut = ms.buffers.saplings
    rt = stand.reference_trees
    next_osid = next_osite_id(stand)

    for layer in range(10):
        for spe_name, internal_species in UT_SPECIES_FIELDS:
            s = getattr(ut[0][layer], spe_name)

            try: # NOTE: Unnecessary try-except?
                if float(s.year) == -1.0:
                    continue
            except TypeError:
                continue

            for cat_code, _ in UT_CATEGORIES:
                stems = float(getattr(s, f"f_{cat_code}", 0.0) or 0.0)
                if stems <= 0.0:
                    continue

                osid_raw = getattr(s, f"osid_{cat_code}", 0.0)
                osid = int(osid_raw)

                if osid <= 0:
                    osid = next_osid
                    next_osid += 1
                    setattr(s, f"osid_{cat_code}", float(osid))

                idx = _find_sapling_reference_tree_index(rt, osid)
                if idx is None:
                    identifier, tree_number = new_reference_tree_identity(stand)
                    storey = _storey_from_layer(stand, layer)
                else:
                    identifier = str(rt.identifier[idx])
                    tree_number = int(rt.tree_number[idx])

                    existing_storey = int(rt.storey[idx])
                    if existing_storey >= 0:
                        storey = existing_storey
                    else:
                        storey = _storey_from_layer(stand, layer)

                row = _build_reference_tree_update(identifier=identifier,
                                                   tree_number=tree_number,
                                                   osid=osid,
                                                   species=internal_species,
                                                   category_code="0",  # Small tree
                                                   stems_per_ha=stems,
                                                   origin_raw=float(getattr(s, f"N_{cat_code}", -1.0) or -1.0),
                                                   height=float(getattr(s, f"h_{cat_code}", 0.0) or 0.0),
                                                   diameter=float(getattr(s, f"d_{cat_code}", 0.0) or 0.0),
                                                   age=float(getattr(s, f"age_{cat_code}", 0.0) or 0.0),
                                                   age13=float(getattr(s, f"age13_{cat_code}", 0.0) or 0.0),
                                                   basal_area=float(getattr(s, f"g_{cat_code}", 0.0) or 0.0),
                                                   volume=float(getattr(s, f"v_{cat_code}", 0.0) or 0.0),
                                                   storey=storey,
                                                   )

                if idx is None:
                    rt.create(row)
                else:
                    rt.update(row, idx)


def _prune_promoted_sapling_reference_trees(stand: ForestStand) -> None:
    """
    Delete old sapling RFs if SID exists in YP vector.
    """
    ms = stand.motti_state
    rt = stand.reference_trees

    if ms is None or ms.yp is None or rt.size == 0:
        return

    yp_strata: set[int] = set()
    for i in range(ms.ntrees):
        t = ms.yp[0][i]
        sid = int(t.sid)
        if sid > 0:
            yp_strata.add(sid)

    if not yp_strata:
        return

    delete_idx: list[int] = []
    for i in range(rt.size):
        if not bool(rt.sapling[i]):
            continue

        sid = int(rt.stratum[i])
        if sid > 0 and sid in yp_strata:
            delete_idx.append(i)

    if delete_idx:
        rt.delete(np.array(delete_idx, dtype=int))


def _prune_reference_trees_not_in_yp(stand: ForestStand) -> None:
    """
    Keep only ReferenceTrees that have a live in the YP vector.
    Used after Motti4Init init.
    """
    rt = stand.reference_trees
    ms = stand.motti_state

    if rt.size == 0:
        return

    live_yp: set[tuple[int, int]] = set()
    if ms is not None and ms.yp is not None:
        for i in range(ms.ntrees):
            t = ms.yp[0][i]
            sid = int(t.sid)
            tree_id = int(t.id)
            if sid > 0 and tree_id > 0:
                live_yp.add((sid, tree_id))

    delete_idx: list[int] = []
    for i in range(rt.size):
        sid = int(rt.stratum[i])
        try: # NOTE: Unnecessary try-except?
            tree_number = int(rt.tree_number[i])
        except (TypeError, ValueError):
            tree_number = -1

        if sid <= 0 or tree_number <= 0 or (sid, tree_number) not in live_yp:
            delete_idx.append(i)

    if delete_idx:
        rt.delete(np.array(delete_idx, dtype=int))

# reconcile_reference_trees_from_motti
def reconcile_reference_trees_from_motti(stand: ForestStand, *, init_mode: bool = False) -> None:
    sync_yp_to_reference_trees(stand)
    _prune_promoted_sapling_reference_trees(stand)

    if init_mode:
        _prune_reference_trees_not_in_yp(stand)

    sync_ut_to_reference_trees(stand)
    prune_reference_trees_not_in_motti(stand)


def _refresh_reference_trees_from_motti_after_yp_change(stand: ForestStand) -> None:
    """
    Rebuild Motti internal state after yp edits, run grow(step=0) and
    then synchronize ReferenceTrees from yp/ut.
    """
    ms = stand.motti_state
    if ms is None or ms.yp is None or ms.buffers is None:
        return

    Motti4DLL.grow_with_state(ms, step=0)

    reconcile_reference_trees_from_motti(stand)


def _reduce_motti_yp_by_removed_reference_trees(stand: ForestStand, removed_f: npt.NDArray[np.float64]) -> bool:
    ms = stand.motti_state
    trees = stand.reference_trees
    if ms is None or ms.yp is None or trees.size == 0:
        return False

    changed = False
    for idx, delta in enumerate(removed_f):
        if delta <= 0.0:
            continue
        if bool(trees.sapling[idx]):
            continue

        sid = int(trees.stratum[idx])
        if sid <= 0:
            continue

        tree_number = int(trees.tree_number[idx])
        if tree_number <= 0:
            continue

        for i in range(ms.ntrees):
            t = ms.yp[0][i]
            if int(t.sid) != sid:
                continue
            if int(t.id) != tree_number:
                continue

            new_f = max(t.f - delta, 0.0)
            if new_f != t.f:
                t.f = new_f
                changed = True
            break

    return changed


def apply_motti_yp_reduction_from_removed_reference_trees(stand: ForestStand,
                                                          removed_f: np.ndarray,
                                                          *,
                                                          refresh: bool = True,
                                                          ) -> bool:
    """
    Generic helper for treatments that reduce tree amounts.
    This helper maps removed trees to the yp vector via the shared stratum/sid,
    optionally runs a zero-step Motti refresh, and synchronizes ReferenceTrees
    back from the refreshed Motti state.
    """
    changed = _reduce_motti_yp_by_removed_reference_trees(stand, removed_f)
    if changed and refresh:
        _refresh_reference_trees_from_motti_after_yp_change(stand)
    return changed

# reconcile_reference_trees_from_motti
def _collect_live_motti_keys(stand: ForestStand) -> set[tuple[str, int, int | None]]:
    live: set[tuple[str, int, int | None]] = set()

    ms = stand.motti_state
    if ms is None:
        return live

    if ms.yp is not None:
        for i in range(ms.ntrees):
            t = ms.yp[0][i]

            sid = int(t.sid)
            tree_id = int(t.id)

            if sid > 0 and tree_id > 0:
                live.add(("yp", sid, tree_id))

    if ms.buffers is not None:
        ut = ms.buffers.saplings
        for layer in range(10):
            for spe_name, _ in UT_SPECIES_FIELDS:
                s: Motti4SaplingStratum = getattr(ut[0][layer], spe_name)
                for cat_code, _ in UT_CATEGORIES:
                    stems = float(getattr(s, f"f_{cat_code}"))
                    if stems <= 0.0:
                        continue
                    osid = int(getattr(s, f"osid_{cat_code}"))
                    if osid > 0:
                        live.add(("ut", osid, None))

    return live


def prune_reference_trees_not_in_motti(stand: ForestStand) -> None:

    rt = stand.reference_trees
    if rt.size == 0:
        return

    live = _collect_live_motti_keys(stand)
    delete_idx = []

    for i, value in enumerate(rt.stratum):
        sid = int(value)
        if sid <= 0:
            continue

        key: tuple[str, int, int | None]
        if bool(rt.sapling[i]):
            key = ("ut", sid, None)
        else:
            try:
                tree_number = int(rt.tree_number[i])
            except (TypeError, ValueError):
                tree_number = -1
            key = ("yp", sid, tree_number)

        if key not in live:
            delete_idx.append(i)

    if delete_idx:
        rt.delete(np.array(delete_idx, dtype=int))
