import difflib
import os
from pathlib import Path
from typing import Iterable, Any

from lukefi.metsi.data.model import ForestStand


SNAP_DIR = Path(__file__).resolve().parent / "snapshots"


def _enum_value(x: Any) -> Any:
    return getattr(x, "value", x)


def _safe_str(x: Any) -> str:
    """Stable stringification for diffs."""
    if x is None:
        return "None"
    if isinstance(x, tuple):
        return "(" + ", ".join(_safe_str(v) for v in x) + ")"
    return str(x)


def stand_to_snapshot_row(s: ForestStand) -> list[str]:
    """
    Produce a stable stand row.
    Store enum values as ints.
    """
    lat = s.geo_location[0] if s.geo_location else None
    lon = s.geo_location[1] if s.geo_location else None
    h = s.geo_location[2] if s.geo_location else None
    crs = s.geo_location[3] if s.geo_location else None

    row = [
        "stand",
        s.identifier,
        _safe_str(s.year),
        _safe_str(s.area),
        _safe_str(s.area_weight),
        _safe_str(lat),
        _safe_str(lon),
        _safe_str(h),
        _safe_str(crs),
        _safe_str(s.degree_days),
        _safe_str(_enum_value(s.owner_category)),
        _safe_str(_enum_value(s.land_use_category)),
        _safe_str(_enum_value(s.soil_peatland_category)),
        _safe_str(_enum_value(s.site_type_category)),
        _safe_str(s.tax_class_reduction),
        _safe_str(s.tax_class),
        _safe_str(_enum_value(s.drainage_category)),
        _safe_str(s.drainage_year),
        _safe_str(s.fertilization_year),
        _safe_str(s.soil_surface_preparation_year),
        _safe_str(s.regeneration_area_cleaning_year),
        _safe_str(s.development_class),
        _safe_str(s.artificial_regeneration_year),
        _safe_str(s.young_stand_tending_year),
        _safe_str(s.cutting_year),
        _safe_str(s.forestry_centre_id),
        _safe_str(s.forest_management_category),
        _safe_str(s.method_of_last_cutting),
        _safe_str(s.municipality_id),
        _safe_str(s.fra_category),
        _safe_str(s.auxiliary_stand),
        _safe_str(s.area_weight_factors[0] if s.area_weight_factors else None),
        _safe_str(s.area_weight_factors[1] if s.area_weight_factors else None),
        _safe_str(s.stand_id),
        _safe_str(s.basal_area),
        _safe_str(s.ds_main_tree_species_biological_age),
        _safe_str(_enum_value(s.main_tree_species_dominant_storey)),
        _safe_str(s.region),
        _safe_str(s.peatland_type),
        _safe_str(s.drained_peatland_type),
        _safe_str(s.under_storey),
        _safe_str(s.over_storey),
    ]
    return row


def stands_to_snapshot_lines(stands: Iterable[ForestStand]) -> list[str]:
    """
    Deterministic text snapshot:
      - Stand line
      - Stratum lines
      - Tree lines
    Uses vector_model.as_internal_csv_row for trees & strata.
    """
    rows: list[list[str]] = []
    for s in stands:
        rows.append(stand_to_snapshot_row(s))

        if s.tree_strata is not None:
            for idx, _ in sorted(
                enumerate(s.tree_strata.identifier),
                key=lambda pair: str(pair[1]),
            ):
                rows.append(s.tree_strata.as_internal_csv_row(idx))

        if s.reference_trees is not None:
            for idx, _ in sorted(
                enumerate(s.reference_trees.identifier),
                key=lambda pair: str(pair[1]),
            ):
                rows.append(s.reference_trees.as_internal_csv_row(idx))

    out_lines: list[str] = []
    for r in rows:
        out_lines.append(",".join(r))
    return out_lines


def _assert_snapshot_text(testcase, *, snap_path: Path, actual_text: str, label: str) -> None:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    update = os.environ.get("UPDATE_MELA_SNAPSHOTS", "") == "1"

    if update:
        snap_path.write_text(actual_text, encoding="utf-8", newline="")
        return

    expected_text = snap_path.read_text(encoding="utf-8")
    if expected_text != actual_text:
        diff = "\n".join(
            difflib.unified_diff(
                expected_text.splitlines(),
                actual_text.splitlines(),
                fromfile=f"expected:{snap_path.name}",
                tofile="actual",
                lineterm="",
            )
        )
        testcase.fail(
            f"Snapshot mismatch for '{label}'.\n\n"
            f"If the change is intended, run: pytest --update-snapshots to regenerate references.\n\n"
            f"Diff:\n{diff}\n"
        )


def assert_snapshot(testcase, *, name: str, stands: Iterable[ForestStand]) -> None:
    """
    Compare current stand snapshot to file.
    """
    snap_path = SNAP_DIR / f"{name}.csv"
    actual_lines = stands_to_snapshot_lines(stands)
    actual_text = "\n".join(actual_lines) + "\n"
    _assert_snapshot_text(testcase, snap_path=snap_path, actual_text=actual_text, label=name)


def assert_file_snapshot(testcase, *, snapshot_name: str, actual_file: str | Path) -> None:
    """
    Compare an exported file against a text snapshot file under tests/data/snapshots.
    """
    actual_path = Path(actual_file)
    actual_text = actual_path.read_text(encoding="utf-8")
    snap_path = SNAP_DIR / snapshot_name
    _assert_snapshot_text(
        testcase,
        snap_path=snap_path,
        actual_text=actual_text,
        label=snapshot_name,
    )
