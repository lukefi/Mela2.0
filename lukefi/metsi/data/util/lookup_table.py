from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Mapping, Sequence, TypeVar

import csv

T = TypeVar("T")  # e.g. ForestStand


@dataclass(frozen=True)
class LookupTable(Generic[T]):
    """
    Generic CSV-backed lookup.

    Assumptions (simple version):
      - key_columns are column names in the CSV.
      - Those same names must exist as attributes on the stand
        (e.g. CSV has 'degree_days' -> stand must have stand.degree_days).
      - Optionally, per-column transform functions can be provided.
        If present, we call transform[column](stand.<column>) before matching.
        If not present, we use stand.<column> raw.

      - CSV value_column is returned, and cast with value_cast (default: int).
    """

    csv_path: str
    key_columns: Sequence[str]
    value_column: str
    transforms: Mapping[str, Callable[[Any], Any]] | None = None
    value_cast: Callable[[str], Any] = int

    def __call__(self, stand: T) -> Any:
        # Build key_values from stand attributes (optionally transformed)
        key_values: Dict[str, Any] = {}
        for col in self.key_columns:
            if not hasattr(stand, col):
                raise AttributeError(
                    f"LookupTable {self.csv_path!r}: stand has no attribute {col!r}"
                )

            raw = getattr(stand, col)
            if self.transforms and col in self.transforms:
                raw = self.transforms[col](raw)

            key_values[col] = raw

        row = _find_matching_row(self.csv_path, key_values)

        # Get and cast the value column
        try:
            raw_value = row[self.value_column]
        except KeyError as e:
            csv_p = Path(self.csv_path).resolve()
            raise ValueError(
                f"Lookup CSV {csv_p} is missing value column {self.value_column!r}."
            ) from e

        try:
            return self.value_cast(raw_value)
        except Exception as e:
            raise ValueError(
                f"Could not convert value {raw_value!r} from column {self.value_column!r} "
                f"in CSV {self.csv_path!r} using {self.value_cast}."
            ) from e


# --- Internal helpers ----------------------------------------------------


@lru_cache(maxsize=None)
def _load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", encoding="utf8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _coerce_int(value: Any, key: str) -> int:
    try:
        return int(value)
    except Exception as e:
        raise ValueError(
            f"Could not convert {key}={value!r} to int when reading lookup table."
        ) from e


def _find_matching_row(csv_path: str, key_values: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Find a *single* row in csv_path where all key columns match key_values.

    - All key values are compared as ints (simple version).
      If you need strings later, we can generalize this.
    """
    csv_p = Path(csv_path).resolve()
    rows = _load_rows(csv_p)

    if not rows:
        raise ValueError(f"Lookup CSV {csv_p} has no data rows.")

    # Ensure all keys exist as CSV columns
    missing_cols = [col for col in key_values.keys() if col not in rows[0]]
    if missing_cols:
        raise ValueError(
            f"CSV {csv_p} is missing required key column(s) {missing_cols!r}."
        )

    candidates: List[Dict[str, Any]] = []

    for row in rows:
        match = True
        for col, key_val in key_values.items():
            row_val = _coerce_int(row[col], col)
            key_val_int = _coerce_int(key_val, col)
            if row_val != key_val_int:
                match = False
                break
        if match:
            candidates.append(row)

    if not candidates:
        raise ValueError(
            f"No matching row in CSV {csv_p} for keys: "
            + ", ".join(f"{k}={v!r}" for k, v in key_values.items())
        )

    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous rows in CSV {csv_p} for keys: "
            + ", ".join(f"{k}={v!r}" for k, v in key_values.items())
        )

    return candidates[0]
