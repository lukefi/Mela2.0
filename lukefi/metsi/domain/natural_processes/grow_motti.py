import numpy as np
from lukefi.metsi.domain.natural_processes.motti_util import reconcile_reference_trees_from_motti
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.data.enums.internal import LandUseCategory, Storey
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.util import update_stand_growth
from lukefi.metsi.domain.natural_processes.natural_process_wrapper import natural_process_transition
from lukefi.metsi.core.collected_data import OpTuple
from lukefi.metsi.core.exceptions import MetsiException
from lukefi.metsi.forestry.storey import calc_tree_basal_areas, calculate_storey_mean_heights, should_use_ba_for_storey


@natural_process_transition
def grow_motti_fn(input_: ForestStand, step: int = 5) -> OpTuple[ForestStand]:
    """
    Motti grow:
      - Builds DLL input from FDM and runs growth
      - Prunes trees with stems_per_ha < 1.0 after update
    """

    stand = input_

    if stand.motti_state is None:
        raise MetsiException("Missing Motti state initialization.")

    trees = stand.reference_trees

    if stand.land_use_category is not None and stand.land_use_category >= LandUseCategory.WASTE_LAND:
        # Can these even be nan? Is the condition necessary?
        base_d = np.nan_to_num(trees.breast_height_diameter, nan=0.0)
        base_h = np.nan_to_num(trees.height, nan=0.0)
        base_f = np.nan_to_num(trees.stems_per_ha, nan=0.0)
        update_stand_growth(stand, base_d, base_h, base_f, step, False)
        return stand, []

    state = stand.motti_state
    state.yy.year = stand.relative_year
    state.yy.step = step

    Motti4DLL.grow_with_state(state, step=step)

    stand.year = (stand.year or 0) + step

    reconcile_reference_trees_from_motti(stand, init_mode=False)

    # Handle new trees' storeys --------------------------------------------------------------------------

    # Pre-calculate basal area for all trees
    trees.basal_area = calc_tree_basal_areas(trees.breast_height_diameter)

    storeys = np.unique(trees.storey)
    has_dominant_storey = Storey.DOMINANT in storeys

    # Merge new trees into DOMINANT, UNDER or OVER storey
    if not has_dominant_storey:
        # Mark new trees as DOMINANT
        trees.storey[trees.storey == Storey.UNSET] = Storey.DOMINANT
    else:
        # Check if should merge new trees into DOMINANT
        mean_heights = calculate_storey_mean_heights(trees, {Storey.DOMINANT, Storey.UNSET})
        diff = mean_heights[Storey.DOMINANT] - mean_heights[Storey.UNSET]
        if abs(diff) < 5.0:
            # Merge into DOMINANT
            trees.storey[trees.storey == Storey.UNSET] = Storey.DOMINANT
        else:
            # Merge into UNDER
            trees.storey[trees.storey == Storey.UNSET] = Storey.UNDER

    # Merge existing storeys ----------------------------------------------------------------------------

    storeys = np.unique(trees.storey)
    mean_heights = calculate_storey_mean_heights(trees, {Storey.DOMINANT, Storey.UNDER, Storey.OVER})

    dominant_storey_mean_height = mean_heights[Storey.DOMINANT]

    if Storey.UNDER in storeys:
        if abs(dominant_storey_mean_height - mean_heights[Storey.UNDER]) < 5.0:
            # Merge UNDER into DOMINANT
            trees.storey[trees.storey == Storey.UNDER] = Storey.DOMINANT

    if Storey.OVER in storeys:
        if abs(mean_heights[Storey.OVER] - dominant_storey_mean_height) < 5.0:
            # Merge OVER into DOMINANT
            trees.storey[trees.storey == Storey.OVER] = Storey.DOMINANT

    return stand, []
