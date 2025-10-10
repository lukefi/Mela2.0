from __future__ import annotations
from pathlib import Path
from typing import Literal
import numpy as np

Area = Literal[1, 2, 3, 4]

def _read_4x5(path: Path) -> np.ndarray:
    """
    Reads a simple 4x5 grid (ints) ignoring comment lines starting with '*'.
    Returns shape (site=4, spe=5).
    """
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("*"):
            continue
        parts = s.split()
        # accept only rows that look like 5 numbers
        nums = [float(x) for x in parts if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
        if len(nums) == 5:
            rows.append(nums)
        if len(rows) == 4:
            break
    if len(rows) != 4:
        raise ValueError(f"Could not parse 4x5 table from {path.name}")
    return np.array(rows, dtype=float)

def load_regeneration_tables(base_dir: str | Path):
    base = Path(base_dir)
    dia_map: dict[Area, np.ndarray] = {}
    age_map: dict[Area, np.ndarray] = {}

    # Area 1 files present now
    p_d = base / "min_regeneration_diameter.txt"
    p_a = base / "min_regeneration_age.txt"
    try:
        dia_map[1] = _read_4x5(p_d)
    except FileNotFoundError:
        ...
    try:
        age_map[1] = _read_4x5(p_a)
    except FileNotFoundError:
        ...

    return dia_map, age_map

def site_to_index(site: int) -> int:
    if site <= 2:
        return 0
    if site == 3:
        return 1
    if site == 4:
        return 2
    return 3

def domspe_to_index5(dom_spe: int) -> int:
    # 1=pine, 2=spruce, 3=raudus (silver birch), 4=hies (downy birch), else -> 5 (other deciduous)
    if dom_spe <= 4:
        return dom_spe - 1
    if dom_spe == 7:
        return 0  # pine
    return 4  # other deciduous
