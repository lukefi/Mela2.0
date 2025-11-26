import numpy as np
import numpy.typing as npt

from lukefi.metsi.data.model import TreeSpecies
from lukefi.metsi.data.vector_model import ReferenceTrees


def yearly_diameter_growth_by_species(
    spe: TreeSpecies,
    d: npt.NDArray[np.float64],
    h: npt.NDArray[np.float64],
    biological_age_aggregate: float,
    d13_aggregate: float,
    height_aggregate: float,
    dominant_height: np.floating,
    basal_area_total: float
) -> npt.NDArray[np.float64]:
    """ Model source: Acta Forestalia Fennica 163 """
    if spe == TreeSpecies.PINE:
        growth_percent = np.exp(5.4625
                                - 0.6675 * np.log(biological_age_aggregate)
                                - 0.4758 * np.log(basal_area_total)
                                + 0.1173 * np.log(d13_aggregate)
                                - 0.9442 * np.log(dominant_height)
                                - 0.3631 * np.log(d)
                                + 0.7762 * np.log(h))
    else:
        growth_percent = np.exp(6.9342
                                - 0.8808 * np.log(biological_age_aggregate)
                                - 0.4982 * np.log(basal_area_total)
                                + 0.4159 * np.log(d13_aggregate)
                                - 0.3865 * np.log(height_aggregate)
                                - 0.6267 * np.log(d)
                                + 0.1287 * np.log(h))
    return growth_percent


def yearly_height_growth_by_species(
    spe: TreeSpecies,
    d: npt.NDArray[np.float64],
    h: npt.NDArray[np.float64],
    biological_age_aggregate: float,
    d13_aggregate: float,
    height_aggregate: float,
    basal_area_total: float
) -> npt.NDArray[np.float64]:
    """ Model source: Acta Forestalia Fennica 163 """
    if spe == TreeSpecies.PINE:
        growth_percent = np.exp(5.4636
                                - 0.9002 * np.log(biological_age_aggregate)
                                + 0.5475 * np.log(d13_aggregate)
                                - 1.1339 * np.log(h))
    else:
        growth_percent = (12.7402
                          - 1.1786 * np.log(biological_age_aggregate)
                          - 0.0937 * np.log(basal_area_total)
                          - 0.1434 * np.log(d13_aggregate)
                          - 0.8070 * np.log(height_aggregate)
                          + 0.7563 * np.log(d)
                          - 2.0522 * np.log(h))
    return growth_percent


def grow_diameter_and_height(trees: ReferenceTrees,
                             step: int = 5) -> tuple[npt.NDArray[np.float64],
                                                     npt.NDArray[np.float64]]:
    """
    Diameter and height growth for trees with height > 1.3 meters. Based on Acta Forestalia Fennica 163.
    Vector data implementation.
    """
    if trees.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    ds = trees.breast_height_diameter.copy()
    hs = trees.height.copy()

    for s in range(step):
        bigh = np.extract(hs >= 1.3, hs)
        if bigh.size > 0:
            hdom = np.median(bigh)
            gs = trees.stems_per_ha * np.pi * (0.01 * 0.5 * ds)**2
            g = np.sum(gs)
            species = np.unique(trees.species)
            for spe in species:
                gg = np.sum(gs, where=trees.species == spe)
                ag = np.sum((trees.biological_age + s) * gs, where=trees.species == spe) / gg
                dg = np.sum(ds * gs, where=trees.species == spe) / gg
                hg = np.sum(hs * gs, where=trees.species == spe) / gg

                condition = (trees.species == spe) & (hs >= 1.3)

                pd = yearly_diameter_growth_by_species(spe,
                                                       ds[condition],
                                                       hs[condition],
                                                       ag,
                                                       dg,
                                                       hg,
                                                       hdom,
                                                       g) / 100
                ph = yearly_height_growth_by_species(spe,
                                                     ds[condition],
                                                     hs[condition],
                                                     ag,
                                                     dg,
                                                     hg,
                                                     g) / 100
                ds[condition] *= (1 + pd)
                hs[condition] *= (1 + ph)
        hs[hs < 1.3] += 0.3
        ds[(ds == 0) & (hs >= 1.3)] = 1.0
    return ds, hs
