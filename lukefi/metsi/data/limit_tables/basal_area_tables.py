from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import re
import numpy as np

# --- Now supports areas 1..4 ---
Area = Literal[1, 2, 3, 4]
SoilInd = Literal[1, 2]
SiteInd = Literal[1, 2, 3, 4]
HInd = Literal[1,2,3,4,5,6,7,8,9]
SpeInd = Literal[1,2,3,4]

@dataclass(frozen=True)
class BAInstructionTables:
    before_upper: dict[Area, np.ndarray]  # (2,4,9,4)
    after_lower: dict[Area, np.ndarray]   # (2,4,9,4)

# Accept both “Vatkg” and “Vatg” (area 4 files use Vatg)
_SECTION_START = re.compile(
    r"^\*(OMT|MT|VT|CT)\s*$|^\*(Rhtg|Mtkg|Ptkg|Vatkg|Vatg)\s*$", re.I
)
_NUMBER_ROW = re.compile(r"^\s*(-?\d+(\.\d+)?\s+)+-?\d+(\.\d+)?\s*$")
_SITE_KEY_MINERAL = ["OMT", "MT", "VT", "CT"]
_SITE_KEY_PEAT    = ["Rhtg", "Mtkg", "Ptkg", "Vatkg", "Vatg"]  # accept both; map to the same index

def _read_matrix_from_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.full((2, 4, 9, 4), np.nan, dtype=float)
    if not path.exists():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    soil_block: Optional[int] = None
    site_idx: Optional[int] = None
    row_cursor = 0

    for raw in lines:
        s = raw.strip()

        if s.upper().startswith("*KANGASMAAT"):
            soil_block = 0
            continue
        if s.upper().startswith("*TURVEMAA"):
            soil_block = 1
            continue

        m = _SECTION_START.match(s)
        if m:
            tag = (m.group(1) or m.group(2) or "").strip()
            if soil_block is None:
                continue
            if soil_block == 0:
                try:
                    site_idx = _SITE_KEY_MINERAL.index(tag.upper())
                except ValueError:
                    site_idx = None
            else:
                # normalize Vatg -> Vatkg
                tagn = "Vatkg" if tag == "Vatg" else tag
                try:
                    # restrict to first 4 peat keys (Rhtg,Mtkg,Ptkg,Vatkg)
                    site_idx = ["Rhtg", "Mtkg", "Ptkg", "Vatkg"].index(tagn)
                except ValueError:
                    site_idx = None
            row_cursor = 0
            continue

        if soil_block is not None and site_idx is not None and _NUMBER_ROW.match(s):
            vals = [float(x) for x in s.split()]
            if len(vals) != 9:
                continue
            if row_cursor < 4:
                arr[soil_block, site_idx, :, row_cursor] = vals
                row_cursor += 1

    return arr, arr.copy()

def _area_from_filename(name: str) -> Optional[int]:
    if name.endswith("4.txt"):
        return 4
    if name.endswith("3.txt"):
        return 3
    if name.endswith("2.txt"):
        return 2
    if name.endswith(".txt"):
        return 1
    return None

def load_ba_instruction_tables(base_dir: str | Path) -> BAInstructionTables:
    base = Path(base_dir)
    before = {}
    after = {}
    for fname in [
        "basal_area_instructions_before_thinning.txt",
        "basal_area_instructions_before_thinning2.txt",
        "basal_area_instructions_before_thinning3.txt",
        "basal_area_instructions_before_thinning4.txt",
        "basal_area_instructions_after_thinning.txt",
        "basal_area_instructions_after_thinning2.txt",
        "basal_area_instructions_after_thinning3.txt",
        "basal_area_instructions_after_thinning4.txt",
    ]:
        p = base / fname
        area = _area_from_filename(fname)
        if area is None:
            continue
        try:
            arr, _ = _read_matrix_from_txt(p)
        except FileNotFoundError:
            continue
        if "before" in fname:
            before[area] = arr
        else:
            after[area] = arr

    if not before:
        raise RuntimeError("No ‘before thinning’ basal-area tables found.")
    if not after:
        raise RuntimeError("No ‘after thinning’ basal-area tables found.")
    return BAInstructionTables(before_upper=before, after_lower=after)

# Index helpers (unchanged)
def site_to_index(site: int) -> SiteInd:
    if site <= 2:
        return 1
    if site == 3:
        return 2
    if site == 4:
        return 3
    return 4

def soil_to_index(soil: int) -> SoilInd:
    return 1 if soil == 1 else 2

def hgm_to_index(hgm: float) -> HInd:
    h_ind = int(round((float(hgm) - 8.0000001) * 0.5))
    return 1 if h_ind < 1 else 9 if h_ind > 9 else h_ind  # type: ignore

def domspe_to_index(dom_spe: int) -> SpeInd:
    if dom_spe <= 4:
        return dom_spe
    if dom_spe == 7:
        return 1
    return 4