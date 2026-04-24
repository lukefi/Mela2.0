import math
import pathlib

from lukefi.metsi.data.formats.nfi.vmi9_builder import VMI9Builder
from lukefi.metsi.data.formats.nfi.vmi_const import (
    VMI9_STAND_INDICES_ESUOMI,
    VMI9_STAND_INDICES_PSUOMI,
    VMI9_STAND_COMMON,
)
from lukefi.metsi.data.formats.nfi import vmi_util
from lukefi.metsi.data.formats.util import parse_int


def test_vmi9_kasittelyluokka():
    base = pathlib.Path(__file__).resolve().parent.parent

    rows_path = base / "resources" / "vmi9_esim_kuviot.txt"
    exp_path = base / "resources" / "vmi9_esim_kasittelyluokat.txt"

    rows = [ln.rstrip("\n") for ln in rows_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    expected = [float(ln.strip()) for ln in exp_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # VMI9 row_type is at fixed position 13
    stand_rows = [r for r in rows if len(r) > 13 and r[13] == "1"]

    assert len(stand_rows) == len(expected), f"stand_rows={len(stand_rows)} expected={len(expected)}"

    for row, exp in zip(stand_rows, expected):
        idx = VMI9Builder._select_stand_indices(row)
        src = vmi_util.generate_source_data(idx, row)
        got = VMI9Builder._determine_forest_management_category(src)
        assert math.isclose(got, exp, rel_tol=0.0, abs_tol=1e-9), (got, exp)
