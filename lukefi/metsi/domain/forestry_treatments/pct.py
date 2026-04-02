from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.treatment import Treatment
from lukefi.metsi.domain.natural_processes.grow_motti_dll import (
    sync_ut_to_reference_trees,
    sync_yp_to_reference_trees,
    prune_reference_trees_not_in_motti,
)


def pct_fn(input_: ForestStand, /, **operation_parameters) -> OpTuple[ForestStand]:
    """
    Motti-only sapling treatment .

    Required parameters:
      remaining_n: int   # target remaining stem count after PCT

    """
    stand = input_

    ms = getattr(stand, "motti_state", None)
    if ms is None or ms.buffers is None:
        raise MetsiException(
            "Motti PCT requested but stand has no initialized motti_state. "
            "Use Motti transition / bootstrap so state exists before this event."
        )

    remaining_n = int(operation_parameters["remaining_n"])
    if remaining_n <= 0:
        raise MetsiException("Parameter 'remaining_n' must be > 0")

    ms.ntrees = ms.dll.pct_with_state(
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

    return stand, []


pct = Treatment(pct_fn, "pct")
