from lukefi.metsi.data.conversion.internal2motti import convert_fertilization_type
from lukefi.metsi.data.enums.internal import FertilizationType
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.motti_util import (
    sync_ut_to_reference_trees,
    sync_yp_to_reference_trees,
    prune_reference_trees_not_in_motti,
)
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.core.treatment import Treatment
from lukefi.metsi.core.collected_data import OpTuple
from lukefi.metsi.core.exceptions import MetsiException


def mineral_soils_fertilization_fn(stand: ForestStand,
                                   /,
                                   fertilization_type: FertilizationType | None = None,
                                   amount_n: float | None = None,
                                   phosphorus: bool = False) -> OpTuple[ForestStand]:
    """
    Mineral soils fertilization treatment function that applies fertilization to a forest stand.

    Usable also with Motti as transition, but for Motti use case needs the motti related parameters.

    Parameters in Motti use case:
    ----------
    fertilization_type : FertilizationType | None
        Fertilization type code passed to Motti.
    amount_n : float | None
        Nitrogen amount for the Motti treatment.
    phosphorus : bool, default False
        Whether phosphorus is included in the fertilization.

    Updates the stand's motti_state and sets the fertilization year to the current year in ForestStand.
    """
    if stand.motti_state is not None:
        ms = stand.motti_state

        if fertilization_type is None:
            raise MetsiException("Fertilization parameter 'fertilization_type' is required for Motti fertilization")
        if amount_n is None:
            raise MetsiException("Fertilization parameter 'amount_n' is required for Motti fertilization")

        Motti4DLL.mineral_soils_fertilization_with_state(
            ms.yy,
            ms.yp,
            ms.ntrees,
            ms.buffers,
            ftype=convert_fertilization_type(fertilization_type).value,
            amount_n=amount_n,
            phosphorus=int(phosphorus)
        )

        # Keep Python-side vectors aligned in case the DLL updated state immediately.
        sync_yp_to_reference_trees(stand)
        sync_ut_to_reference_trees(stand)
        prune_reference_trees_not_in_motti(stand)

    stand.fertilization_year = stand.year

    return stand, []


mineral_soils_fertilization = Treatment(mineral_soils_fertilization_fn, "mineral_soils_fertilization")
