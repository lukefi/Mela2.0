from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.motti_util import sync_ut_to_reference_trees, sync_yp_to_reference_trees
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.treatment import Treatment
from lukefi.metsi.domain.natural_processes.motti_util import (
    prune_reference_trees_not_in_motti,
)


def mineral_soils_fertilization_fn(stand: ForestStand,
                                   /,
                                   ftype: int | None = None,
                                   amount_n: float | None = None,
                                   phosphorus: int = 0
                               ) -> OpTuple[ForestStand]:
    """
    Motti-only mineral-soils fertilization treatment.

    Parameters
    ----------
    ftype : int
        Fertilization type code passed through to Motti.
    amount_n : float
        Nitrogen amount. Alias: amountN
    bool_phosphorus : int
        0/1 flag indicating whether phosphorus is included.
        Aliases: boolPhosporus, phosphorus
    """

    ms = stand.motti_state
    if ms is None or ms.buffers is None:
        raise MetsiException(
            "Motti mineral-soils fertilization requested but stand has no initialized motti_state. "
        )

    if ftype is None:
        raise MetsiException("Fertilization parameter 'ftype' is required")

    if amount_n is None:
        raise MetsiException("Fertilization parameter 'amount_n' (or amountN) is required")

    _ = Motti4DLL.mineral_soils_fertilization_with_state(
        ms.yy,
        ms.yp,
        ms.ntrees,
        ms.buffers,
        ftype=ftype,
        amount_n=amount_n,
        bool_phosphorus=int(bool(phosphorus)),
    )

    # Keep Python-side vectors aligned in case the DLL updated state immediately.
    sync_yp_to_reference_trees(stand)
    sync_ut_to_reference_trees(stand)
    prune_reference_trees_not_in_motti(stand)

    stand.fertilization_year = stand.year

    return stand, []


mineral_soils_fertilization = Treatment(mineral_soils_fertilization_fn, "mineral_soils_fertilization")
