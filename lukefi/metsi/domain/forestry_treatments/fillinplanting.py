from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.conversion.internal2motti import convert_species
from lukefi.metsi.data.enums.internal import TreeSpecies
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.treatment import Treatment
from lukefi.metsi.domain.natural_processes.motti_util import (
    next_osite_id, sync_ut_to_reference_trees, sync_yp_to_reference_trees)
from lukefi.metsi.domain.natural_processes.motti_util import (
    prune_reference_trees_not_in_motti,
)


def fillinplanting_fn(stand: ForestStand,
                      /,
                      species: TreeSpecies = TreeSpecies.TREELESS,
                      stems_per_ha: float = 0.0,
                      osite_id: int | None = None) -> OpTuple[ForestStand]:
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

    ms = stand.motti_state
    if ms is None or ms.buffers is None:
        raise MetsiException(
            "Motti FillinPlanting requested but stand has no initialized motti_state. "
            "Use Motti transition / bootstrap so state exists before this event."
        )

    if species <= 0:
        raise MetsiException("FillinPlanting requires parameter 'species' (internal TreeSpecies code)")

    if stems_per_ha <= 0.0:
        raise MetsiException("FillinPlanting parameter 'stems_per_ha' must be > 0")

    if osite_id is None:
        osite_id = next_osite_id(stand)
    if osite_id <= 0:
        raise MetsiException("FillinPlanting parameter 'osite_id' must be > 0")

    rspe = convert_species(species)

    ms.ntrees = Motti4DLL.fillin_planting_with_state(
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
