from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.treatment import Treatment
from lukefi.metsi.domain.natural_processes.util import next_osite_id
from lukefi.metsi.domain.natural_processes.grow_motti_dll import (
    species_to_motti,
    sync_ut_to_reference_trees,
    sync_yp_to_reference_trees,
    prune_reference_trees_not_in_motti,
)


def fillinplanting_fn(input_: ForestStand, /, **operation_parameters) -> OpTuple[ForestStand]:
    """
    Motti-only fill-in planting treatment.

    Parameters
    ----------
    species : int
        Internal TreeSpecies code for the planted species.
    stems_per_ha : float
        Number of planted saplings per hectare.
    osite_id : int, optional
        Target saplin stratum id. If omitted, a new id is allocated.

    Note
    -----
    Motti DLL expects species in Motti coding (rspe), planting amount as a float
    (num), and the planted cohort id (ositeID).
    """
    stand = input_

    ms = stand.motti_state
    if ms is None or ms.buffers is None:
        raise MetsiException(
            "Motti FillinPlanting requested but stand has no initialized motti_state. "
            "Use Motti transition / bootstrap so state exists before this event."
        )

    species = int(operation_parameters.get("species", operation_parameters.get("rspe", 0)))
    if species <= 0:
        raise MetsiException("FillinPlanting requires parameter 'species' (internal TreeSpecies code)")

    stems_per_ha = float(operation_parameters.get("stems_per_ha", operation_parameters.get("num", 0.0)))
    if stems_per_ha <= 0.0:
        raise MetsiException("FillinPlanting parameter 'stems_per_ha' must be > 0")

    osite_id = int(operation_parameters.get("osite_id", next_osite_id(stand)))
    if osite_id <= 0:
        raise MetsiException("FillinPlanting parameter 'osite_id' must be > 0")

    rspe = species_to_motti(species)

    ms.ntrees = ms.dll.fillin_planting_with_state(
        ms.yy,
        ms.yp,
        int(ms.ntrees),
        ms.buffers,
        rspe=rspe,
        num=stems_per_ha,
        osite_id=osite_id,
    )

    sync_yp_to_reference_trees(stand)
    sync_ut_to_reference_trees(stand)
    prune_reference_trees_not_in_motti(stand)

    stand.artificial_regeneration_year = stand.year

    return stand, []


fillinplanting = Treatment(fillinplanting_fn, "fillinplanting")
