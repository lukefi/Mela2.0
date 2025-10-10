from __future__ import annotations
import numpy as np
import numpy.typing as npt
from lukefi.metsi.data.vector_model import ReferenceTrees

class SelectionData:
    """
    Read-only proxy for select_units.
    Provides:
      - g: per-tree basal area (m2) = pi/40000 * d^2  (d in cm)
      - v: TEMP ordering proxy, mapped to diameter
      - R aliases: d, age, age13, spe, manag_cat
    """
    def __init__(self, trees: ReferenceTrees):
        self._t = trees
        self.size = trees.size
        self.g: npt.NDArray[np.float64] = np.pi / 40000.0 * (trees.breast_height_diameter ** 2)
        self.v: npt.NDArray[np.float64] = trees.breast_height_diameter
        # R alias views (no copies)
        self.d = trees.breast_height_diameter
        self.age = getattr(trees, "biological_age", np.full_like(self.d, np.nan))
        self.age13 = getattr(trees, "breast_height_age", np.full_like(self.d, np.nan))
        self.spe = getattr(trees, "species", None)
        self.manag_cat = getattr(trees, "management_category", None)

    def __getitem__(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        if hasattr(self._t, name):
            return getattr(self._t, name)
        raise AttributeError(name)