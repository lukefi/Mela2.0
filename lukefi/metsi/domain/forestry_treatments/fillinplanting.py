import numpy as np

from lukefi.metsi.data.conversion.internal2motti import convert_species
from lukefi.metsi.data.enums.internal import Storey, TreeSpecies
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.motti_util import (
    sync_ut_to_reference_trees,
    prune_reference_trees_not_in_motti
)
from lukefi.metsi.forestry.aggregates_internal import calc_mean_height, calc_tree_basal_areas
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.core.exceptions import MetsiException
from lukefi.metsi.core.treatment import Treatment
from lukefi.metsi.core.collected_data import OpTuple


def fillinplanting_fn(stand: ForestStand,
                      /,
                      species: TreeSpecies = TreeSpecies.TREELESS,
                      stems_per_ha: float = 0.0) -> OpTuple[ForestStand]:
    """
    Motti-only filling planting treatment.

    Parameters
    ----------
    species:
        Internal TreeSpecies code for the planted species.
    stems_per_ha:
        Number of planted saplings per hectare.
    Note
    -----
    Motti handels stratum id generation
    """

    ms = stand.motti_state
    if ms is None or ms.buffers is None:
        raise MetsiException(
            "Fillingplanting with Motti failed as motti_state has not been initialized."
        )

    if species <= 0:
        raise MetsiException("FillinPlanting requires parameter 'species' (internal TreeSpecies code)")

    if stems_per_ha <= 0.0:
        raise MetsiException("FillinPlanting parameter 'stems_per_ha' must be > 0")

    filled_species = convert_species(species)

    ms.ntrees = Motti4DLL.fillin_planting_with_state(
        ms,
        filled_species=filled_species,
        stems_per_ha=stems_per_ha,
    )

    sync_ut_to_reference_trees(stand)
    prune_reference_trees_not_in_motti(stand)

    # Handle storeys ----------------------------------------------------------------------------------------------

    trees = stand.reference_trees

    if np.any(trees.storey == Storey.DOMINANT):
        # Pre-calculate basal area for all trees
        trees.basal_area = calc_tree_basal_areas(trees.breast_height_diameter)

        diff = calc_mean_height(trees, trees.storey == Storey.DOMINANT) - \
            calc_mean_height(trees, trees.storey == Storey.UNSET)
        if abs(diff) < 5.0:
            # Merge new trees into DOMINANT
            storey_for_new_trees = Storey.DOMINANT
        else:
            # Merge new trees into UNDER
            storey_for_new_trees = Storey.UNDER
    else:
        # No previous DOMINANT storey exists, promote new trees to DOMINANT
        storey_for_new_trees = Storey.DOMINANT

    if not trees.storey.flags.writeable:
        trees.storey = np.copy(trees.storey)

    trees.storey[trees.storey == Storey.UNSET] = storey_for_new_trees

    return stand, []


fillinplanting = Treatment(fillinplanting_fn, "fillinplanting")
