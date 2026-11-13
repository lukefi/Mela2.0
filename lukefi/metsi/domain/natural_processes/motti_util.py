from typing import Any

import numpy as np
import numpy.typing as npt

from lukefi.metsi.data.conversion.internal2motti import convert_storey
from lukefi.metsi.data.enums.internal import Origin, Storey, TreeCategory, TreeManagementCategory, TreeSpecies
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.motti.motti_types import Motti4SaplingStratum, Motti4Saplings, MottiState
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
    """ Synchronize Motti tree vector into stand reference trees. """

    if stand.motti_state is None:
        raise ValueError("Cannot sync Motti tree vector into stand reference trees." \
        "MottiState is not initialized for the stand.")

    ms = stand.motti_state
    yp = ms.yp
    rt = stand.reference_trees

    rts_for_create: list[dict[str, Any]] = []
    rts_for_update: dict[str, list[Any]] = {
        "stems_per_ha": [],
        "height": [],
        "breast_height_diameter": [],
        "biological_age": [],
        "breast_height_age": [],
        "basal_area": [],
        "volume": []
    }
    idx_for_update: list[int] = []
    itree_number: int = np.max(rt.tree_number).item() if rt.size > 0 else 0

    for i in range(ms.ntrees):
        t = yp[0][i]
        yp_tree_id = int(t.id)
        yp_stratum_id = str(int(t.sid))

        _matching_stratum = rt.stratum == yp_stratum_id
        yp_not_in_rt = (np.any(_matching_stratum) and
                        np.any(rt.sapling[_matching_stratum]))

        if yp_not_in_rt:
            itree_number += 1
            t.id = itree_number
            rts_for_create.append(
                {
                    "identifier": f"{stand.identifier}-{itree_number}-tree",
                    "tree_number": itree_number,
                    "species": TreeSpecies(int(t.spe)),
                    "breast_height_diameter": t.d13,
                    "height": t.h,
                    "breast_height_age": t.age13,
                    "biological_age": t.age,
                    "stems_per_ha": t.f,
                    "origin": Origin(int(t.snt) - 1),
                    "management_category": TreeManagementCategory.NO_RESTRICTION,
                    "basal_area": (t.ba / 10000.0),  # cm2 -> m2
                    "volume": t.vol,
                    "stratum": int(t.sid)
                }
            )
        else:
            idx_for_update.append(np.where(rt.tree_number == yp_tree_id)[0][0].item())
            rts_for_update["stems_per_ha"].append(t.f)
            rts_for_update["height"].append(t.h)
            rts_for_update["breast_height_diameter"].append(t.d13)
            rts_for_update["biological_age"].append(t.age)
            rts_for_update["breast_height_age"].append(t.age13)
            rts_for_update["basal_area"].append(t.ba / 10000.0) # cm2 -> m2
            rts_for_update["volume"].append(t.vol)

    if rts_for_create:
        rt.create(rts_for_create)
    if rts_for_update:
        rt.update_many(rts_for_update, idx_for_update)


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
    return {
        "identifier": identifier,
        "tree_number": tree_number,
        "stratum": str(osid),
        "species": species,
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

            try:
                # If Motti stratum year is -1, it means that the stratum is not yet active and should be skipped.
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
                    # If osid is not set, allocate a new one and update the Motti sapling stratum
                    osid = next_osid
                    next_osid += 1
                    setattr(s, f"osid_{cat_code}", float(osid))

                idx = _find_sapling_reference_tree_index(rt, osid)
                existing_storey = None
                if idx is None:
                    # Create a new tree
                    (identifier,
                    tree_number) = new_reference_tree_identity(stand)
                else:
                    # Update a existing tree
                    identifier = rt.identifier[idx].item()
                    tree_number = rt.tree_number[idx].item()
                    existing_storey = rt.storey[idx].item()

                row = _build_reference_tree_update(identifier=identifier,
                                                   tree_number=tree_number,
                                                   osid=osid,
                                                   species=internal_species,
                                                   category_code=TreeCategory.SMALL_TREE.value,
                                                   stems_per_ha=stems,
                                                   origin_raw=getattr(s, f"N_{cat_code}", -1.0),
                                                   height=getattr(s, f"h_{cat_code}", 0.0),
                                                   diameter=getattr(s, f"d_{cat_code}", 0.0),
                                                   age=getattr(s, f"age_{cat_code}", 0.0),
                                                   age13=getattr(s, f"age13_{cat_code}", 0.0),
                                                   basal_area=getattr(s, f"g_{cat_code}", 0.0),
                                                   volume=getattr(s, f"v_{cat_code}", 0.0),
                                                   storey=existing_storey or Storey.UNSET.value
                                                   )

                if idx is None:
                    rt.create(row)
                else:
                    rt.update(row, idx)


def _prune_promoted_sapling_reference_trees(stand: ForestStand) -> None:
    """
    Delete sapling reference trees that exist in the YP vector.
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


def reconcile_reference_trees_from_motti(stand: ForestStand) -> None:
    sync_yp_to_reference_trees(stand)
    _prune_promoted_sapling_reference_trees(stand)
    sync_ut_to_reference_trees(stand)
    prune_reference_trees_not_in_motti(stand)


def _update_motti_aggregates(stand: ForestStand) -> None:
    """
    Rebuild Motti internal state after yp edits, run grow(step=0) and
    then synchronize ReferenceTrees from yp/ut.
    """
    ms = stand.motti_state
    if ms is None or ms.yp is None or ms.buffers is None:
        return
    Motti4DLL.grow_with_state(ms, step=0)
    reconcile_reference_trees_from_motti(stand)


def _update_motti_yp_after_cutting(stand: ForestStand, removed_stems: npt.NDArray[np.float64]) -> bool:
    """ Updates Motti tree vector (a.k.a yp) from given removed stems. Returns True if any updates were made.

        The update is done reducing the amount of stems_per_ha in the yp vector by the amount of removed stems.
    """
    ms = stand.motti_state
    trees = stand.reference_trees
    if ms is None or ms.yp is None or trees.size == 0:
        return False

    changed = False
    for idx, delta in enumerate(removed_stems):
        if delta <= 0.0:
            # No removed trees for this reference tree, skip
            continue

        if bool(trees.sapling[idx]):
            # Sapling reference tree, skip.
            continue

        sid = int(trees.stratum[idx])
        if sid <= 0:
            # No stratum, skip.
            continue

        tree_number = int(trees.tree_number[idx])
        if tree_number <= 0:
            # No tree number, skip.
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


def _update_ut_after_cutting(ut: Motti4Saplings, saplings: ReferenceTrees) -> bool:
    """ Updates Motti sapling vector (a.k.a ut) stems_per_ha values with given sapling data.
        Returns True if any updates were made.
    """
    changed = False
    if saplings.size == 0:
        return changed

    for i in range(saplings.size):
        spe_rt = saplings.species[i].item()
        stratum_rt = float(saplings.stratum[i].item())
        f_rt = saplings.stems_per_ha[i].item()
        is_found = False
        for j in range(10):
            ut_layer = ut[0][j]
            for ut_spe_name, species in UT_SPECIES_FIELDS:
                if species != spe_rt:
                    # No species match, skip to next species.
                    continue
                ut_strata = getattr(ut_layer, ut_spe_name)
                for cat_code, _ in UT_CATEGORIES:
                    if (ut_strata.year <= 0.0 and ut_strata.age <= 0.0):
                        # sapling strata inactive, skip.
                        continue
                    osid_ut = getattr(ut_strata, f"osid_{cat_code}")
                    if osid_ut != stratum_rt:
                        # reference tree stratum does not match Motti sapling stratum, skip.
                        continue
                    # Found matching Motti sapling stratum, update stems_per_ha
                    setattr(ut_strata, f"f_{cat_code}", f_rt)
                    changed = True
                    is_found = True
                    break
                if is_found:
                    break
            if is_found:
                break
    return changed


def _update_motti_trees_after_cutting(stand: ForestStand,
                                      removed_stems: npt.NDArray[np.float64]
                                      ) -> bool:
    """ Updates Motti trees and saplings after cutting.
        Returns True if either yp or ut trees were updated.
    """
    assert stand.motti_state # for lintter
    changed = False

    changed |= _update_motti_yp_after_cutting(
        stand, removed_stems
    )
    changed |= _update_ut_after_cutting(
        ut=stand.motti_state.buffers.saplings,
        saplings=stand.reference_trees[stand.reference_trees.sapling]
    )
    return changed


def update_motti_state_after_cutting(stand: ForestStand,
                                    removed_stems: npt.NDArray[np.float64],
                                    ) -> bool:
    """ Used after cutting treatments to update Motti state.

        Note: This function is called after the cutting treatment has been applied to the stand \
            and the reference trees have been updated accordingly.
    
        stand: contains up to date tree and sapling data.

        removed_stems: array of removed stems per tree.

        return: True if Motti state was updated, False if no updates were made.
    """
    is_changed = False
    is_changed = _update_motti_trees_after_cutting(stand, removed_stems)
    if is_changed:
        _update_motti_aggregates(stand)
    return is_changed


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
                    if stems <= 0.0 or s.year == -1:
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


def split_and_update_motti_yp_tree(ms: MottiState,
                                   new_tree_id: int,
                                   tree_number: int,
                                   stratum: int,
                                   f_new: float,
                                   f_upd: float,
                                   new_storey: Storey | None = None):
    """
    Creates a copy with a new storey of a existing yp tree and updates existing tree stem count.

    An existing yp tree is searched based on `tree_number` and `stratum` values.
    After existing tree is found a new tree is created from the existing tree.
    The new tree is given differing values for id, story and stem count.  

    Finally the stem count of the existing tree is updated.
    """
    next_idx = ms.ntrees
    for i in range(ms.ntrees):
        t = ms.yp[0][i]
        if int(t.id) == tree_number and int(t.sid) == stratum:
            # split yp tree and "mark" with attributes
            ms.yp[0][next_idx].id = float(new_tree_id)
            ms.yp[0][next_idx].sid = t.sid
            ms.yp[0][next_idx].spe = t.spe
            ms.yp[0][next_idx].d13 = t.d13
            ms.yp[0][next_idx].h = t.h
            ms.yp[0][next_idx].age13 = t.age13
            ms.yp[0][next_idx].snt = t.snt
            ms.yp[0][next_idx].storie = t.storie if new_storey is None else float(convert_storey(new_storey))
            ms.yp[0][next_idx].f = float(f_new)
            # update existing yp tree
            t.f = float(f_upd)
    ms.ntrees += 1
