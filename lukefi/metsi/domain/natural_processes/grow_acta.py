from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.util import update_stand_growth
from lukefi.metsi.forestry.naturalprocess.grow_acta import grow_diameter_and_height
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.treatment import Treatment


def grow_acta_fn(input_: ForestStand, /, **operation_parameters) -> OpTuple[ForestStand]:
    step = operation_parameters.get('step', 5)
    stand = input_
    if stand.reference_trees.size == 0:
        stand.year += step
        return stand, []
    diameters, heights = grow_diameter_and_height(stand.reference_trees, step)
    stems = stand.reference_trees.stems_per_ha
    update_stand_growth(stand, diameters, heights, stems, step)
    return stand, []


grow_acta = Treatment(grow_acta_fn, "grow_acta")
