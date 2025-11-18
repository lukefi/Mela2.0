from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import csv
import json

@dataclass
class MinStemsConfig:
    mode: str                 # "per_area_files" | "single_file"
    site_key: str
    species_key: str
    value_key: str
    area_key: Optional[str] = None
    files_by_area: Dict[str, str] | None = None
    csv: Optional[str] = None

@lru_cache(maxsize=None)
def _csv_has_multiple_areas(csv_path: str, area_key: str | None) -> bool:
    """
    Returns True if the CSV has an area_key column and more than one distinct
    non-empty value in that column. Otherwise False.

    This lets us automatically decide whether to use area_group in lookups.
    """
    if not area_key:
        return False

    path = Path(csv_path)
    with path.open("r", encoding="utf8", newline="") as f:
        reader = csv.DictReader(f)
        values = set()
        for row in reader:
            if area_key in row and row[area_key] != "":
                values.add(row[area_key])
                if len(values) > 1:
                    return True  # early exit: we know it's multi-area

    # 0 or 1 distinct values -> treat as single-area (ignore area in indexing)
    return False

@lru_cache(maxsize=None)
def _load_config(config_path: str) -> MinStemsConfig:
    """
    Load and cache JSON configuration that tells how to index the CSV table.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf8") as f:
        raw = json.load(f)

    mode = raw.get("mode")
    files_by_area = raw.get("files_by_area")
    csv_file = raw.get("csv")

    # Auto-deduce mode if not explicitly given
    if mode is None:
        if files_by_area:
            mode = "per_area_files"
        elif csv_file:
            mode = "single_file"
        else:
            raise ValueError(
                f"Min stems config {config_path!r} must define either 'files_by_area' or 'csv'."
            )

    return MinStemsConfig(
        mode=mode,
        site_key=raw.get("site_key", "site_group"),
        species_key=raw.get("species_key", "species_group"),
        value_key=raw.get("value_key", "min_stems"),
        area_key=raw.get("area_key"),
        files_by_area=files_by_area,
        csv=csv_file,
    )


@lru_cache(maxsize=None)
def _load_rows(csv_path: str) -> List[Dict[str, Any]]:
    """
    Read a CSV into a list of dict rows (cached).
    """
    path = Path(csv_path)
    with path.open("r", encoding="utf8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _coerce_int(value: Any, key: str) -> int:
    try:
        return int(value)
    except Exception as e:
        raise ValueError(
            f"Could not convert {key}={value!r} to int when reading min stems table."
        ) from e


def lookup_min_stems(
    config_path: str,
    *,
    area_group: int | None,
    site_group: int,
    species_group: int,
) -> int:
    """
    Main API:
      - config_path: path to JSON metadata file
      - area_group, site_group, species_group: 1-based indices

    Raises ValueError if no matching row (or ambiguous rows) are found.
    """
    cfg = _load_config(config_path)

    if cfg.mode == "per_area_files":
        if not cfg.files_by_area:
            raise ValueError(
                f"Config {config_path!r} in 'per_area_files' mode must define 'files_by_area'."
            )

        # If only one file is configured, ignore area_group and always use it.
        if len(cfg.files_by_area) == 1:
            csv_path = next(iter(cfg.files_by_area.values()))
        else:
            if area_group is None:
                raise ValueError(
                    f"Config {config_path!r} expects area_group, but none was given."
                )
            key = str(area_group)
            try:
                csv_path = cfg.files_by_area[key]
            except KeyError as e:
                raise ValueError(
                    f"No CSV file defined for area_group={area_group} in {config_path!r}."
                ) from e

        rows = _load_rows(csv_path)
        return _lookup_in_rows(
            rows,
            cfg,
            area_group=None,   # encoded by file choice already
            site_group=site_group,
            species_group=species_group,
        )

    if cfg.mode == "single_file":
        if not cfg.csv:
            raise ValueError(
                f"Config {config_path!r} in 'single_file' mode must define 'csv'."
            )
        rows = _load_rows(cfg.csv)
        return _lookup_in_rows(
            rows,
            cfg,
            area_group=area_group,
            site_group=site_group,
            species_group=species_group,
        )

    raise ValueError(
        f"Unsupported min stems config mode {cfg.mode!r} in {config_path!r}."
    )


def _lookup_in_rows(
    rows: Iterable[Dict[str, Any]],
    cfg: MinStemsConfig,
    *,
    area_group: int | None,
    site_group: int,
    species_group: int,
) -> int:
    """
    Filter CSV rows by (area_group?, site_group, species_group) and return min_stems.
    """
    candidates: List[Dict[str, Any]] = []

    for row in rows:
        try:
            row_site = _coerce_int(row[cfg.site_key], cfg.site_key)
            row_species = _coerce_int(row[cfg.species_key], cfg.species_key)
        except KeyError as e:
            raise ValueError(
                f"Missing '{e.args[0]}' column in min stems CSV; expected at least "
                f"{cfg.site_key!r}, {cfg.species_key!r} and {cfg.value_key!r}."
            ) from e

        if cfg.area_key and area_group is not None:
            try:
                row_area = _coerce_int(row[cfg.area_key], cfg.area_key)
            except KeyError as e:
                raise ValueError(
                    f"Config requires area_key={cfg.area_key!r} but CSV has no such column."
                ) from e
            if row_area != area_group:
                continue

        if row_site == site_group and row_species == species_group:
            candidates.append(row)

    if not candidates:
        raise ValueError(
            "No matching row in min stems CSV for "
            f"area_group={area_group}, site_group={site_group}, species_group={species_group}."
        )
    if len(candidates) > 1:
        raise ValueError(
            "Ambiguous min stems rows in CSV for "
            f"area_group={area_group}, site_group={site_group}, species_group={species_group}."
        )

    row = candidates[0]
    try:
        value = row[cfg.value_key]
    except KeyError as e:
        raise ValueError(
            f"Min stems CSV missing value column {cfg.value_key!r}"
        ) from e

    return _coerce_int(value, cfg.value_key)
