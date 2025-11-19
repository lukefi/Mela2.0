from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import csv
import json


@dataclass
class MinStemsConfig:
    """
    Configuration for the min-stems lookup.

    The config only says:
      - which columns are used as a key (key_columns)
      - which column contains the value (value_key)

    The actual CSV file is chosen per event.
    """
    key_columns: List[str]
    value_key: str


@lru_cache(maxsize=None)
def _load_config(config_path: str) -> MinStemsConfig:
    """
    Load and cache JSON configuration that tells how to index the CSV table.
    """
    cfg_path = Path(config_path).resolve()
    with cfg_path.open("r", encoding="utf8") as f:
        raw = json.load(f)

    key_columns = raw.get("key_columns")
    if not key_columns or not isinstance(key_columns, list):
        raise ValueError(
            f"Min stems config {config_path!r} must define 'key_columns' as a non-empty list."
        )

    value_key = raw.get("value_key", "min_stems")

    return MinStemsConfig(
        key_columns=[str(k) for k in key_columns],
        value_key=str(value_key),
    )


@lru_cache(maxsize=None)
def _load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Read a CSV into a list of dict rows (cached).
    """
    with csv_path.open("r", encoding="utf8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _coerce_int(value: Any, key: str) -> int:
    try:
        return int(value)
    except Exception as e:
        raise ValueError(
            f"Could not convert {key}={value!r} to int when reading min stems table."
        ) from e


def min_stems_lookup(
    config_path: str,
    csv_path: str,
    key_values: Dict[str, Any],
) -> int:
    """
    Main API for callers (e.g. FirstThinningMineralSoils):

      - config_path: path to JSON metadata file (absolute or relative)
      - csv_path:    path to the concrete CSV file for this event
                     (absolute, or relative to the JSON file directory)
      - key_values:  mapping from column name -> key value
                     e.g. {"site_group": 1, "species_group": 2, "dd_group": 3}

    Behaviour:

      - JSON is read once and cached.
      - Each CSV path is read once and cached.
      - Raises ValueError if:
          * required key is missing in key_values,
          * required column is missing in CSV,
          * no matching row is found,
          * more than one matching row is found,
          * the value column is missing or non-integer.
    """
    cfg = _load_config(config_path)

    cfg_path = Path(config_path).resolve()
    csv_p = Path(csv_path)

    # If CSV path is relative, resolve it relative to the config file location
    if not csv_p.is_absolute():
        csv_p = (cfg_path.parent / csv_p).resolve()

    rows = _load_rows(csv_p)

    missing_keys = [k for k in cfg.key_columns if k not in key_values]
    if missing_keys:
        raise ValueError(
            f"Missing key value(s) {missing_keys} for min stems lookup; "
            f"required keys are {cfg.key_columns}."
        )

    candidates: List[Dict[str, Any]] = []

    for row in rows:
        try:
            # Check all key columns match
            match = True
            for col in cfg.key_columns:
                if col not in row:
                    raise ValueError(
                        f"CSV {csv_p} is missing required key column {col!r} "
                        f"defined in config {config_path!r}."
                    )

                row_val = _coerce_int(row[col], col)
                key_val = _coerce_int(key_values[col], col)

                if row_val != key_val:
                    match = False
                    break

            if not match:
                continue

            candidates.append(row)

        except KeyError as e:
            raise ValueError(
                f"CSV {csv_p} is missing column {e.args[0]!r}."
            ) from e

    if not candidates:
        raise ValueError(
            f"No matching row in min stems CSV {csv_p} for keys: "
            + ", ".join(f"{k}={key_values.get(k)!r}" for k in cfg.key_columns)
        )

    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous rows in min stems CSV {csv_p} for keys: "
            + ", ".join(f"{k}={key_values.get(k)!r}" for k in cfg.key_columns)
        )

    row = candidates[0]
    try:
        value = row[cfg.value_key]
    except KeyError as e:
        raise ValueError(
            f"Min stems CSV {csv_p} is missing value column {cfg.value_key!r}."
        ) from e

    return _coerce_int(value, cfg.value_key)
