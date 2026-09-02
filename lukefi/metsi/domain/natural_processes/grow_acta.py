from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.natural_process_wrapper import natural_process_transition
from lukefi.metsi.domain.natural_processes.util import update_stand_growth
from lukefi.metsi.forestry.naturalprocess.grow_acta import grow_diameter_and_height
from lukefi.metsi.core.collected_data import OpTuple


@natural_process_transition
def grow_acta_fn(input_: ForestStand, step: int = 5) -> OpTuple[ForestStand]:
    stand = input_
    if stand.reference_trees.size == 0:
        stand.year += step
        return stand, []
    diameters, heights = grow_diameter_and_height(stand.reference_trees, step)
    stems = stand.reference_trees.stems_per_ha
    update_stand_growth(stand, diameters, heights, stems, step)
    return stand, []
