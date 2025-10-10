from __future__ import annotations
from typing import Optional, TypedDict
from pathlib import Path
import numpy as np

from lukefi.metsi.data.limit_tables.basal_area_tables import (
    load_ba_instruction_tables,
    site_to_index as site_to_index_ba,
    soil_to_index, hgm_to_index, domspe_to_index as domspe_to_index4,
)

from lukefi.metsi.data.limit_tables.regeneration_tables import (
    load_regeneration_tables,
    site_to_index as site_to_index_reg,
    domspe_to_index5,
)

from lukefi.metsi.data.limit_tables.stems_after_thinning_tables import (
    load_stems_after_thinning_tables,
    site_to_index as site_to_index_stems,
    domspe_to_index4 as domspe_to_index_stems,
)

_TABLES_BA = None
_TABLES_REG = None      # (dia_map, age_map)
_TABLES_STEMS = None    # (area12, area34)

def _ba_tables():
    global _TABLES_BA
    if _TABLES_BA is None:
        base_dir = Path(__file__).resolve().parent / "txt"
        _TABLES_BA = load_ba_instruction_tables(base_dir)
    return _TABLES_BA

def _reg_tables():
    global _TABLES_REG
    if _TABLES_REG is None:
        base_dir = Path(__file__).resolve().parent / "txt"
        _TABLES_REG = load_regeneration_tables(base_dir)
    return _TABLES_REG

def _stems_tables():
    global _TABLES_STEMS
    if _TABLES_STEMS is None:
        base_dir = Path(__file__).resolve().parent / "txt"
        _TABLES_STEMS = load_stems_after_thinning_tables(base_dir)
    return _TABLES_STEMS

class StandLike(TypedDict, total=False):
    site: int
    soil: int
    Hgm: float
    dom_spe: int
    degree_days: Optional[int]

def _area_from_degree_days(stand: StandLike) -> int:
    """
    Areas (by degree-days):
      1: > 1200
      2: 1000–1200
      3:  900–1000
      4:  < 900
    """
    dd = stand.get("degree_days", None)
    if dd is None:
        return 1
    if dd > 1200:
        return 1
    if dd >= 1000:
        return 2
    if dd >= 900:
        return 3
    return 4

# ---------------- Basal-area instruction limits ----------------

def basal_area_instruction_lower_limit(stand: StandLike, default: float = 16.0) -> float:
    try:
        t = _ba_tables()
        area = _area_from_degree_days(stand)
        soil = soil_to_index(int(stand["soil"])) - 1
        site = site_to_index_ba(int(stand["site"])) - 1
        h    = hgm_to_index(float(stand["Hgm"])) - 1
        spe  = domspe_to_index4(int(stand["dom_spe"])) - 1
        arr = t.after_lower[area]
        val = float(arr[soil, site, h, spe])
        return val if np.isfinite(val) else default
    except Exception:
        return default

def basal_area_instructions_upper_limit(stand: StandLike, default: float = 24.0) -> float:
    try:
        t = _ba_tables()
        area = _area_from_degree_days(stand)
        soil = soil_to_index(int(stand["soil"])) - 1
        site = site_to_index_ba(int(stand["site"])) - 1
        h    = hgm_to_index(float(stand["Hgm"])) - 1
        spe  = domspe_to_index4(int(stand["dom_spe"])) - 1
        arr = t.before_upper[area]
        val = float(arr[soil, site, h, spe])
        return val if np.isfinite(val) else default
    except Exception:
        return default

# ---------------- Regeneration limits (diameter & age) ----------------

def min_regeneration_diameter(stand: StandLike, default: float = 26.0) -> float:
    """
    4×5 table (site collapse 1..4; species 1..5 with 5 = other deciduous).
    Currently only area=1 is in your files; others fallback to default.
    """
    try:
        dia_map, _ = _reg_tables()
        area = _area_from_degree_days(stand)
        if area not in dia_map:
            return default
        site = site_to_index_reg(int(stand["site"]))
        spe5 = domspe_to_index5(int(stand["dom_spe"]))
        val = float(dia_map[area][site, spe5])
        return val if np.isfinite(val) else default
    except Exception:
        return default

def min_regeneration_age(stand: StandLike, default: int = 70) -> int:
    try:
        _, age_map = _reg_tables()
        area = _area_from_degree_days(stand)
        if area not in age_map:
            return default
        site = site_to_index_reg(int(stand["site"]))
        spe5 = domspe_to_index5(int(stand["dom_spe"]))
        val = float(age_map[area][site, spe5])
        return int(val) if np.isfinite(val) else default
    except Exception:
        return default

# ---------------- Min stems after thinning (first thinning rule) ----------------

def min_number_of_stems_after_thinning(stand: StandLike, default: int = 1500) -> int:
    """
    4×4 table; areas 1–2 use file #1, areas 3–4 use file #2.
    Species mapping: 1..4 => 1..4; others fold into birch column (3), per R code.
    """
    try:
        area12, area34 = _stems_tables()
        area = _area_from_degree_days(stand)
        if area in (1, 2):
            grid = area12
        else:
            grid = area34
        if grid is None:
            return default
        site = site_to_index_stems(int(stand["site"]))
        spe4 = domspe_to_index_stems(int(stand["dom_spe"]))
        val = float(grid[site, spe4])
        return int(val) if np.isfinite(val) else default
    except Exception:
        return default