from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.motti_util import sync_ut_to_reference_trees
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.treatment import Treatment
from lukefi.metsi.domain.natural_processes.motti_util import (
    prune_reference_trees_not_in_motti,
)


def seedlingdelay_fn(stand: ForestStand,
                     /,
                     istep: int | None = None) -> OpTuple[ForestStand]:
    """
    Motti-only seedling delay treatment.

    Parameters
    ----------
    istep : int
        Age change in years.
        Positive values increase age, negative values decrease age.

    Notes
    -----
    Motti applies the change only to the last sapling layer and only to
    sapling cohorts whose age is 0 or 1 years.
    """

    ms = stand.motti_state
    if ms is None or ms.buffers is None:
        raise MetsiException(
            "Motti SeedlingDelay requested but stand has no initialized motti_state. "
        )

    if istep is None:
        raise MetsiException("SeedlingDelay parameter 'istep' is required")

    Motti4DLL.seedling_delay_with_state(
        ms.yy,
        ms.buffers,
        istep=istep,
    )

    sync_ut_to_reference_trees(stand)
    prune_reference_trees_not_in_motti(stand)

    return stand, []


seedlingdelay = Treatment(seedlingdelay_fn, "seedlingdelay")
