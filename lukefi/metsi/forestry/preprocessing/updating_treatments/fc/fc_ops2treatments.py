from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.treatment import PredeterminedTreatment
from .fc_cutting import (
    over_storey_cutting,
    first_thinning,
    thinning,
    clearcutting
)
from .fc_regeneration import (
    seeding,
    seeding_pine,
    seeding_spruce,
    seeding_silver_birch,
    seeding_downy_birch,
    planting,
    planting_pine,
    planting_spruce,
    planting_silver_birch,
    planting_downy_birch
)
from .fc_young_stand import (
    mechanical_clearing,
    early_tending,
)

IMPLEMENTED_FC_TREATMENTS: dict[int, PredeterminedTreatment[ForestStand]] = {
    1: over_storey_cutting,      # Ylispuiden poisto
    2: first_thinning,           # Ensiharvennus
    3: thinning,                 # Harvennus
    5: clearcutting,             # Avohakkuu
    200: seeding,                # Kylvö
    201: seeding_pine,           # Männyn kylvö
    202: seeding_spruce,         # Kuusen kylvö
    203: seeding_silver_birch,   # Rauduskoivun kylvö
    204: seeding_downy_birch,    # Hieskoivun kylvö
    300: planting,               # Istutus
    301: planting_pine,          # Männyn istutus
    302: planting_spruce,        # Kuusen istutus
    303: planting_silver_birch,  # Rauduskoivun istutus
    304: planting_downy_birch,   # Hieskoivun istutus
    660: mechanical_clearing,    # Mekaaninen perkaus
    740: early_tending,          # Taimikonhoito
    750: early_tending,          # Nuoren metsän hoito
}
