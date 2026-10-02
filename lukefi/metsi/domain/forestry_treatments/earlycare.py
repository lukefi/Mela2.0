from copy import deepcopy

from lukefi.metsi.core.exceptions import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.collected_data import RemovedTrees
from lukefi.metsi.domain.forestry_treatments.motti_treatment_util import collect_removed_trees
from lukefi.metsi.domain.natural_processes.motti_util import (
    sync_ut_to_reference_trees,
    sync_yp_to_reference_trees,
    prune_reference_trees_not_in_motti,
)
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.core.collected_data import OpTuple
from lukefi.metsi.core.treatment import Treatment


def earlycare_fn(stand: ForestStand, /, imode: int = 0) -> OpTuple[ForestStand]:
    """
    Motti-only early care treatment.

    Parameters
    ----------
    imode : int, optional
        0 = preserve cultivated trees (default)
        1 = also take from cultivated trees if needed

    Returns
    -------
    stand, []
        Stand is updated in-place and synchronized from Motti yp/ut vectors.
    """
    ms = stand.motti_state
    if ms is None or ms.buffers is None:
        raise MetsiException(
            "Motti EarlyCare requested but stand has no initialized motti_state. "
        )

    if imode not in (0, 1):
        raise MetsiException("EarlyCare parameter 'imode' must be 0 or 1")

    original_rts = deepcopy(stand.reference_trees)

    _ = Motti4DLL.earlycare_with_state(
        ms.yy,
        ms.yp,
        ms.ntrees,
        ms.buffers,
        imode=imode,
    )

    # Update ReferenceTrees from Motti vectors
    sync_yp_to_reference_trees(stand)
    sync_ut_to_reference_trees(stand)
    prune_reference_trees_not_in_motti(stand)

    stand.young_stand_tending_year = stand.year

    # Collect removed trees for CollectedData
    cd: list[RemovedTrees] = []
    rmt = RemovedTrees()
    removed_trees = collect_removed_trees(original_rts, stand.reference_trees)
    rmt.removed_trees = removed_trees
    if removed_trees.size > 0:
        cd.append(rmt)

    return stand, cd


earlycare = Treatment(earlycare_fn, "earlycare")
