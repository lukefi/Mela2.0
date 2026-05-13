from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.motti_util import sync_ut_to_reference_trees, sync_yp_to_reference_trees
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.treatment import Treatment
from lukefi.metsi.domain.natural_processes.motti_util import (
    prune_reference_trees_not_in_motti,
)


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

    return stand, []


earlycare = Treatment(earlycare_fn, "earlycare")
