import numpy as np
from lukefi.metsi.domain.natural_processes.motti_util import (
    reconcile_reference_trees_from_motti,
    ensure_state)
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import (
    Motti4DLL,
)
from lukefi.metsi.data.enums.internal import (
    LandUseCategory,
)
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.util import (
    update_stand_growth
)
from lukefi.metsi.domain.natural_processes.natural_process_wrapper import natural_process_transition
from lukefi.metsi.sim.collected_data import OpTuple


@natural_process_transition
def grow_motti_fn(input_: ForestStand, step: int = 5) -> OpTuple[ForestStand]:
    """
    Motti grow:
      - Builds DLL input from FDM and runs growth
      - Prunes trees with stems_per_ha < 1.0 after update
    """

    stand = input_

    rt = stand.reference_trees

    if stand.land_use_category is not None and stand.land_use_category >= LandUseCategory.WASTE_LAND:
        base_d = np.nan_to_num(rt.breast_height_diameter, nan=0.0)
        base_h = np.nan_to_num(rt.height, nan=0.0)
        base_f = np.nan_to_num(rt.stems_per_ha, nan=0.0)
        update_stand_growth(stand, base_d, base_h, base_f, step, False)
        return stand, []

    state = ensure_state(stand, step=step, sim_year=stand.relative_year)
    state.yy.year = stand.relative_year
    state.yy.step = step

    Motti4DLL.grow_with_state(state, step=step)

    stand.year = (stand.year or 0) + step

    reconcile_reference_trees_from_motti(stand, init_mode=False)

    return stand, []
