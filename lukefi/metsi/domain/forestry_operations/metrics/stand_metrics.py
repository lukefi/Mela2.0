# lukefi/metsi/domain/metrics/stand_metrics.py
from __future__ import annotations
from typing import Optional, TypedDict
import numpy as np
import numpy.typing as npt

from lukefi.metsi.data.vector_model import ReferenceTrees

class StandMetrics(TypedDict):
    N: float      # stems/ha
    G: float      # m2/ha
    Dgm: float    # cm (quadratic mean diameter)
    Hgm: float    # m  (basal-area–weighted mean height)
    dom_spe: Optional[int]  # species code with max basal area share (or None)

def compute_stand_metrics(trees: ReferenceTrees) -> StandMetrics:
    """
    Compute N, G, Dgm, Hgm, dom_spe from vector data, without mutating anything.
    Units follow the R side: d in cm, G in m2/ha, h in m.
    """
    f: npt.NDArray[np.float64] = trees.stems_per_ha
    d: npt.NDArray[np.float64] = trees.breast_height_diameter
    h: npt.NDArray[np.float64] = getattr(trees, "height", np.full_like(d, np.nan))
    spe: Optional[npt.NDArray[np.int32]] = getattr(trees, "species", None)

    # basal area per tree (m2): pi/40000 * d^2  (d in cm)
    g = np.pi / 40000.0 * (d ** 2)
    fg = f * g

    N = float(np.nansum(f))
    G = float(np.nansum(fg))
    if G > 0.0:
        Dgm = float(np.nansum(fg * d) / G)  # cm
        Hgm = float(np.nansum(fg * h) / G)  # m
    else:
        Dgm = 0.0
        Hgm = 0.0

    dom_spe_val: Optional[int] = None
    if spe is not None and spe.size > 0 and np.any(~np.isnan(fg)):
        # species with max total basal area
        uniq = np.unique(spe)
        if uniq.size:
            # sum fg by species
            totals = {int(s): float(np.nansum(fg[spe == s])) for s in uniq}
            dom_spe_val = max(totals.items(), key=lambda kv: kv[1])[0] if totals else None

    return {"N": N, "G": G, "Dgm": Dgm, "Hgm": Hgm, "dom_spe": dom_spe_val}
