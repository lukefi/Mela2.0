from copy import deepcopy

from lukefi.metsi.data.enums.motti import MottiSpecies
from lukefi.metsi.data.model import ForestStand, MottiState
from lukefi.metsi.domain.collected_data import RemovedTrees, CollectedData
from lukefi.metsi.domain.natural_processes.motti_util import (
    sync_ut_to_reference_trees,
    sync_yp_to_reference_trees,
    prune_reference_trees_not_in_motti,
)
from lukefi.metsi.domain.forestry_treatments.treatment_util import collect_removed_trees
from lukefi.metsi.forestry.aggregates_internal import calc_tree_basal_areas
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.core.collected_data import OpTuple
from lukefi.metsi.core.treatment import Treatment
from lukefi.metsi.core.exceptions import MetsiException
from lukefi.metsi.forestry.storey import manage_existing_storeys


def pct_fn(stand: ForestStand,
           /,
           remaining_n: list[float] | dict[MottiSpecies, float] | None = None) -> OpTuple[ForestStand]:
    """
    Motti-only sapling treatment.
    
    Relative to pre-commercial thinnig. Also adds naturally regenerated saplings on the stand.

    Optional parameters:
      remaining_n:
          dict or list describing species-wise remaining stems per hectar for motti species
          if not provided Motti resolves based on its guidelines.

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

    trees = stand.reference_trees

    # Pre-calculate basal areas for all trees and handle storey changes
    trees.basal_area = calc_tree_basal_areas(trees.breast_height_diameter)
    manage_existing_storeys(trees)

    cd: list[CollectedData] = []
    rmt = RemovedTrees()
    removed_trees = collect_removed_trees(rts_original, trees)
    rmt.removed_trees = removed_trees
    if removed_trees.size > 0:
        cd.append(rmt)


    return stand, cd


def _resolve_remaining_n(ms: MottiState,
                         remaining_stems_by_species: list[float] | dict[MottiSpecies, float] | None
                        ) -> list[float]:
    """ Prefer parametrized remaining stem counts over Motti pct guidelines """
    if remaining_stems_by_species is not None:
        return _normalize_species_array(remaining_stems_by_species)
    return Motti4DLL.pct_guidelines_with_state(
        ms.yy,
        ms.yp,
        ms.ntrees,
        ms.buffers,
    )


def _normalize_species_array(values: list[float] | dict[MottiSpecies, float]) -> list[float]:
    """
    Normalize caller-provided remaining stem count into a 10-slot list for motti.
    Index 0..8 are motti species, slot index 9 is unused. Values are the remaining stems counts.
    """
    if isinstance(values, dict):
        arr = [0.0] * 10
        for key, stems in values.items():
            idx = int(key)
            if not 1 <= idx <= 9:
                raise MetsiException(f"remaining species must be a value between 1 and 9, got {idx}")
            idx_shift = idx - 1
            arr[idx_shift] = max(float(stems), 0.0)
        return arr

    vals = [float(x) for x in values]
    diff = 10 - len(vals)
    if len(vals) <= 10:
        return [max(v, 0) for v in vals] + [0.0] * diff

    raise MetsiException(
        "Motti does not relate with more than 9 species maximum"
    )


pct = Treatment(pct_fn, "pct", collected_data={RemovedTrees})
