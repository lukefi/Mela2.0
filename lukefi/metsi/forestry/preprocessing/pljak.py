from typing import cast

import pandas as pd
from lukefi.metsi.data.enums.internal import DevelopmentClass, LandUseCategory, StratumRank
from lukefi.metsi.data.enums.vmi import VmiIteration

_spe_proportions: pd.DataFrame
_spe_proportions_loaded = False  # pylint: disable=invalid-name # this is not a constant


def get_spe_proportions(
        land_use_class: LandUseCategory,
        county: int,
        development_class: DevelopmentClass,
        asema: StratumRank,
        dgm: float,
        stems: float,
        spelm: int,
        nfi_iteration: VmiIteration) -> list[float]:
    global _spe_proportions  # pylint: disable=global-statement
    global _spe_proportions_loaded  # pylint: disable=global-statement

    if not _spe_proportions_loaded:
        _spe_proportions = pd.read_csv(f"lukefi/metsi/data/nfi_data/{nfi_iteration.upper()}/pljak_osuuspaino.csv",
                                       sep=' ',
                                       index_col=["speOsNum", "tyyppi", "maakunta"])
        _spe_proportions_loaded = True

    strtype = ""

    if land_use_class == LandUseCategory.SCRUB_LAND:
        strtype = "Kitumaa"
    elif land_use_class == LandUseCategory.FOREST:
        taimikko = (
            (development_class in (
                DevelopmentClass.YOUNG_SEEDLING_STAND,
                DevelopmentClass.ADVANCED_SEEDLING_STAND) and asema in (
                StratumRank.UNPRODUCTIVE_SEEDLINGS,
                StratumRank.DOMINANT_TREE_STOREY)) or (
                asema in (
                    StratumRank.UNDER_STOREY_DEVELOPMENT_CAPABLE,
                    StratumRank.UNDER_STOREY_NOT_DEVELOPMENT_CAPABLE) and stems > 0) or asema == StratumRank.SEEDLING_STRATUM)
        if taimikko and stems >= 3000 and dgm > 0:
            strtype = "MetsaTiheaTaimikko"
        if taimikko and stems < 3000 and dgm > 0:
            strtype = "MetsaHarvaTaimikko"
        if not taimikko and dgm > 0:
            strtype = "MetsaKeskim"
        if not taimikko and dgm > 15:
            strtype = "MetsaVart"

    if (spelm, strtype, county) in _spe_proportions.index:
        proportions_: pd.Series = cast(pd.Series, _spe_proportions.loc[spelm].loc[strtype].loc[county])
        proportions: list[float] = list(proportions_)
    else:
        proportions = [0] * 31

    return proportions
