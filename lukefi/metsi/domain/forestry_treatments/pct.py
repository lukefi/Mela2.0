from copy import deepcopy

import numpy as np

from lukefi.metsi.data.enums.motti import MottiSpecies
from lukefi.metsi.data.model import ForestStand, MottiState
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.domain.collected_data import RemovedTrees
from lukefi.metsi.domain.natural_processes.motti_util import (
    sync_ut_to_reference_trees,
    sync_yp_to_reference_trees,
    prune_reference_trees_not_in_motti,
)
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.core.collected_data import OpTuple
from lukefi.metsi.core.treatment import Treatment
from lukefi.metsi.core.exceptions import MetsiException


def collect_removed_trees(rts_original: ReferenceTrees, rts_modified: ReferenceTrees) -> ReferenceTrees:
    """ Collect trees that have been removed or reduced in stem count. """
    id1 = rts_original.tree_number
    f1 = rts_original.stems_per_ha
    id2 = rts_modified.tree_number
    f2 = rts_modified.stems_per_ha
    # Trees that have have been reduced in stem count
    id_common, idx1, idx2 = np.intersect1d(
        id1,
        id2,
        return_indices=True
    )
    f_diff = f1[idx1] - f2[idx2]
    f_diff_mask = f_diff > 0.0
    id_changed = id_common[f_diff_mask]
    f_diff_changed = f_diff[f_diff_mask]

    # Trees that have been completely removed
    id_removed = np.setdiff1d(id1, id2)

    # Compose the removed trees into a new ReferenceTrees object
    removed_trees: ReferenceTrees = rts_original[np.isin(id1, np.concatenate([id_changed, id_removed]))]
    removed_trees.stems_per_ha[np.isin(removed_trees.tree_number, id_changed)] = f_diff_changed

    return removed_trees


def pct_fn(stand: ForestStand,
           /,
           remaining_n: list[int] | dict[MottiSpecies, int] | None = None) -> OpTuple[ForestStand]:
    """
    Motti-only sapling treatment.

    Preferred parameters:
      remaining_n_by_species:
          dict or list/tuple describing species-wise remaining stem counts
          for species slots 1..9

    """

    ms = stand.motti_state
    if ms is None or ms.buffers is None:
        raise MetsiException(
            "Motti PCT requested but stand has no initialized motti_state. "
            "Use Motti transition / bootstrap so state exists before this event."
        )
    if stand.reference_trees.size == 0:
        return stand, []

    rts_original = deepcopy(stand.reference_trees)

    remaining_n = _resolve_remaining_n(ms, remaining_n)

    ms.ntrees = Motti4DLL.pct_with_state(
        ms.yy,
        ms.yp,
        int(ms.ntrees),
        ms.buffers,
        remaining_n=remaining_n,
    )

    # Keep Python-side vectors aligned with Motti after the treatment.
    sync_yp_to_reference_trees(stand)
    sync_ut_to_reference_trees(stand)
    prune_reference_trees_not_in_motti(stand)
    
    stand.young_stand_tending_year = stand.year

    cd: list[RemovedTrees] = []
    rmt = RemovedTrees()
    removed_trees = collect_removed_trees(rts_original, stand.reference_trees)
    rmt.removed_trees = removed_trees
    if removed_trees.size > 0:
        cd.append(rmt)

    return stand, cd


def _resolve_remaining_n(ms: MottiState, remaining_n: list[int] | dict[MottiSpecies, int] | None) -> list[int]:
    """
    Preferred flow:
      1) ask Motti for guideline array
      2) optionally override or scale it
      3) pass the resulting species-wise array to Motti4PCT
    """
    guidelines = Motti4DLL.pct_guidelines_with_state(
        ms.yy,
        ms.yp,
        ms.ntrees,
        ms.buffers,
    )

    # New preferred parameter: explicit species-wise values
    if remaining_n is not None:
        return _normalize_species_array(remaining_n)

    # Default: use Motti recommendation directly
    return guidelines


def _normalize_species_array(value: list[int] | dict[MottiSpecies, int]) -> list[int]:
    """
    Normalize caller-provided species-wise remaining counts into a 10-slot list.
    Slots 1..9 are species, slot 0 is unused.
    """
    if isinstance(value, dict):
        if all(isinstance(key, MottiSpecies) for key in value.keys()):
            arr = [0] * 10
            for key, stems in value.items():
                idx = int(key)
                if not 1 <= idx <= 9:
                    raise MetsiException(f"remaining_n_by_species index must be 1..9, got {idx}")
                arr[idx] = max(int(stems), 0)
            return arr
        else:
            raise MetsiException(f"all dict keys should correspond to MottiSpecies enum in {value}")

    vals = [int(x) for x in value]
    if len(vals) == 9:
        return [0] + [max(v, 0) for v in vals]
    if len(vals) == 10:
        out = [max(v, 0) for v in vals]
        out[0] = 0
        return out

    raise MetsiException(
        "remaining_n_by_species must be dict or list/tuple of length 9 or 10"
    )


pct = Treatment(pct_fn, "pct")
