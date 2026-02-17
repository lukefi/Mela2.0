import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Mapping, Sequence, TypeVar
import csv

T = TypeVar("T")  # e.g. ForestStand


@dataclass
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

    _index: Dict[tuple[str, ...], str] = field(default_factory=dict, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)
    _load_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _is_it_loaded(self) -> None:
        if self._loaded:
            return

        with self._load_lock:
            if self._loaded:
                return

            csv_p = Path(self.csv_path).resolve()
            idx: Dict[tuple[str, ...], str] = {}

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
            self._loaded = True

    def __call__(self, stand: T) -> Any:
        self._is_it_loaded()

        key_parts: list[str] = []
        debug_pairs: list[tuple[str, Any, Any]] = []

        for col in self.key_columns:
            original = getattr(stand, col)

            if self.transforms and col in self.transforms:
                transformed = self.transforms[col](original)
            else:
                transformed = original

            key_parts.append(str(transformed))
            debug_pairs.append((col, original, transformed))

        try:
            raw_value = self._index[tuple(key_parts)]
        except KeyError as e:
            csv_p = Path(self.csv_path).resolve()

            details = ", ".join(
                f"{col}=original:{orig!r} -> transformed:{trans!r}"
                for col, orig, trans in debug_pairs
            )

            raise ValueError(
                f"No matching row in CSV {csv_p} for keys: {details}"
            ) from e

        try:
            return self.value_cast(raw_value)
        except Exception as e:
            raise ValueError(
                f"Could not convert value {raw_value!r} from column {self.value_column!r} "
                f"in CSV {self.csv_path!r} using {self.value_cast}."
            ) from e
