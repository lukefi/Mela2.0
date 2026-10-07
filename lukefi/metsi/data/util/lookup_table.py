from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import csv


@dataclass(slots=True)
class LookupTable[T, V]:
    """
    Generic CSV-backed lookup.

    Assumptions (simple version):
      - key_columns are column names in the CSV.
      - Those same names must exist as attributes on the stand
        (e.g. CSV has 'degree_days' -> stand must have stand.degree_days).
      - Optionally, per-column transform functions can be provided.
        If present, we call transform[column](stand.<column>) before matching.
        If not present, we use stand.<column> raw.

      - CSV value_column is returned, and cast with value_cast.
    """

    csv_path: str
    key_columns: Sequence[str]
    value_column: str
    key_transforms: Mapping[str, Callable[[Any], Any]] | None
    value_cast: Callable[[str], V]

    _index: dict[tuple[str, ...], str]

    def __init__(self,
                 csv_path: str,
                 key_columns: Sequence[str],
                 value_column: str,
                 value_cast: Callable[[str], V],
                 key_transforms: Mapping[str, Callable[[Any], Any]] | None = None):

        self.csv_path = csv_path
        self.key_columns = key_columns
        self.value_column = value_column
        self.key_transforms = key_transforms
        self.value_cast = value_cast

        csv_p = Path(self.csv_path).resolve()
        idx: dict[tuple[str, ...], str] = {}

        with csv_p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Lookup CSV {csv_p} is missing a header row.")

            required = set(self.key_columns) | {self.value_column}
            missing = [c for c in required if c not in reader.fieldnames]
            if missing:
                raise ValueError(f"CSV {csv_p} is missing required column(s) {missing!r}.")

            row_count = 0
            for row in reader:
                row_count += 1
                key = tuple(str(row[c]) for c in self.key_columns)

                if key in idx:
                    raise ValueError(f"Ambiguous rows in CSV {csv_p} for keys {key}.")

                idx[key] = str(row[self.value_column])

        if row_count == 0:
            raise ValueError(f"Lookup CSV {csv_p} has no data rows.")

        self._index = idx

    def __call__(self, unit: T) -> V:
        key_parts: list[str] = []
        debug_pairs: list[tuple[str, Any, Any]] = []

        for col in self.key_columns:
            original = getattr(unit, col)

            if self.key_transforms and col in self.key_transforms:
                transformed = self.key_transforms[col](original)
            else:
                transformed = original

            key_parts.append(str(transformed))
            debug_pairs.append((col, original, transformed))

        try:
            raw_value = self._index[tuple(key_parts)]

        except KeyError as e:
            csv_p = Path(self.csv_path).resolve()
            details = ", ".join(
                f"{col}=original:{orig!r} -> transformed:{trans!r}" for col, orig, trans in debug_pairs
            )
            raise ValueError(f"No matching row in CSV {csv_p} for keys: {details}") from e

        try:
            return self.value_cast(raw_value)

        except Exception as e:
            raise ValueError(
                f"Could not convert value {raw_value!r} from column {self.value_column!r} "
                f"in CSV {self.csv_path!r} using {self.value_cast}.") from e
