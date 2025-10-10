from __future__ import annotations
from pathlib import Path
from typing import Literal
import numpy as np

Area = Literal[1, 2, 3, 4]

def _read_4x4(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("*"):
            continue
        parts = s.split()
        nums = [float(x) for x in parts if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
        if len(nums) == 4:
            rows.append(nums)
        if len(rows) == 4:
            break
    if len(rows) != 4:
        raise ValueError(f"Could not parse 4x4 table from {path.name}")
    return np.array(rows, dtype=float)

def load_stems_after_thinning_tables(base_dir: str | Path):
    base = Path(base_dir)
    p12 = base / "min_number_of_stems_after_thinning.txt"   # areas 1–2
    p34 = base / "min_number_of_stems_after_thinning2.txt"  # areas 3–4

    area12 = None
    area34 = None
    try:
        area12 = _read_4x4(p12)
    except FileNotFoundError:
        pass
    try:
        area34 = _read_4x4(p34)
    except FileNotFoundError:
        pass

    return area12, area34

def site_to_index(site: int) -> int:
    if site <= 2:
        return 0
    if site == 3:
        return 1
    if site == 4:
        return 2
    return 3

def domspe_to_index4(dom_spe: int) -> int:
    # 1..4 map directly; others fold into birch (3) as in the R helper
    if dom_spe <= 4:
        return dom_spe - 1
    return 2
