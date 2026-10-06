from lukefi.metsi.core.collected_data import OpTuple
from lukefi.metsi.data.enums.internal import DrainageCategory, SoilPeatlandCategory
from lukefi.metsi.data.model import ForestStand


def peatland_drainage_fn(stand: ForestStand) -> OpTuple[ForestStand]:
    stand.drainage_year = stand.year

    if ((stand.drainage_category == DrainageCategory.UNDRAINED_MIRE) or
        (stand.drainage_category == DrainageCategory.UNDRAINED_MINERAL_SOIL_OR_MIRE and
            stand.soil_peatland_category != SoilPeatlandCategory.MINERAL_SOIL)):
        # TODO: Should failing above conditions raise exception? Or should it all be handled with preconditions?
        # TODO: DRAINED_MIRE?
        stand.drainage_category = DrainageCategory.DITCHED_MIRE

    return stand, []
