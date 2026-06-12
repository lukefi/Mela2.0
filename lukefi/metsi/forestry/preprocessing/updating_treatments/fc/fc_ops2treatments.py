from fc_cutting import (
    overStoreyCutting,
    firstThinning,
    thinning,
    clearCutting
)
from fc_regeneration import (
    seeding,
    seedingPine,
    seedingSpruce,
    seedingSilverBirch,
    seedingDownyBirch,
    planting,
    plantingPine,
    plantingSpruce,
    plantingSilverBirch,
    plantingDownyBirch
)
from fc_young_stand import (
    mechanicalClearing,
    earlyTending,
)

FORESTRY_CENTER_TREATMENTS = {
    1: overStoreyCutting,      # Ylispuiden poisto
    2: firstThinning,          # Ensiharvennus
    3: thinning,               # Harvennus
    5: clearCutting,           # Avohakkuu
    200: seeding,              # Kylvö
    201: seedingPine,          # Männyn kylvö
    202: seedingSpruce,        # Kuusen kylvö
    203: seedingSilverBirch,   # Rauduskoivun kylvö
    204: seedingDownyBirch,    # Hieskoivun kylvö
    300: planting,             # Istutus
    301: plantingPine,         # Männyn istutus
    302: plantingSpruce,       # Kuusen istutus
    303: plantingSilverBirch,  # Rauduskoivun istutus
    304: plantingDownyBirch,   # Hieskoivun istutus
    660: mechanicalClearing,   # Mekaaninen perkaus
    740: earlyTending,         # Taimikonhoito
    750: earlyTending,         # Nuoren metsän hoito
}
