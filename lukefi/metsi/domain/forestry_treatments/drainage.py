from lukefi.metsi.core.collected_data import OpTuple
from lukefi.metsi.core.treatment import Treatment
from lukefi.metsi.data.enums.internal import DrainageCategory, SoilPeatlandCategory
from lukefi.metsi.data.model import ForestStand


def drainage_fn(stand: ForestStand) -> OpTuple[ForestStand]:
    stand.drainage_year = stand.year

    if stand.drainage_category == DrainageCategory.UNDRAINED_MIRE:
        stand.drainage_category = DrainageCategory.DITCHED_MIRE

    elif stand.drainage_category == DrainageCategory.UNDRAINED_MINERAL_SOIL_OR_MIRE:
        if stand.soil_peatland_category == SoilPeatlandCategory.MINERAL_SOIL:
            stand.drainage_category = DrainageCategory.DITCHED_MINERAL_SOIL
        else:
            stand.drainage_category = DrainageCategory.DITCHED_MIRE

    elif stand.drainage_category == DrainageCategory.UNDRAINED_MINERAL_SOIL:
        stand.drainage_category = DrainageCategory.DITCHED_MINERAL_SOIL

    return stand, []


peatland_drainage = Treatment(drainage_fn, "drainage", {"drainage", "ditching"})
