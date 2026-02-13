from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Mapping, Sequence, TypeVar
import pandas as pd

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

    # Cached, built once
    _index: Dict[tuple[str, ...], str] = field(default_factory=dict, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    def _is_it_loaded(self) -> None:
        if self._loaded:
            return

        csv_p = Path(self.csv_path).resolve()
        df = pd.read_csv(csv_p, dtype=str)

        if df.empty:
            raise ValueError(f"Lookup CSV {csv_p} has no data rows.")

        missing = [c for c in list(self.key_columns) + [self.value_column] if c not in df.columns]
        if missing:
            raise ValueError(f"CSV {csv_p} is missing required column(s) {missing!r}.")

        # Build dict: (k1,k2,k3) -> value
        idx: Dict[tuple[str, ...], str] = {}
        for _, row in df.iterrows():
            key = tuple(str(row[c]) for c in self.key_columns)
            if key in idx:
                raise ValueError(f"Ambiguous rows in CSV {csv_p} for keys {key}.")
            idx[key] = str(row[self.value_column])

        self._index = idx
        self._loaded = True

    def __call__(self, stand: T) -> Any:
        self._is_it_loaded()

        key_parts: list[str] = []
        for col in self.key_columns:
            raw = getattr(stand, col)
            if self.transforms and col in self.transforms:
                raw = self.transforms[col](raw)
            key_parts.append(str(raw))

        key = tuple(key_parts)

        try:
            raw_value = self._index[key]
        except KeyError as e:
            csv_p = Path(self.csv_path).resolve()
            raise ValueError(
                f"No matching row in CSV {csv_p} for keys: "
                + ", ".join(f"{k}={v!r}" for k, v in zip(self.key_columns, key_parts))
            ) from e

        try:
            return self.value_cast(raw_value)
        except Exception as e:
            raise ValueError(
                f"Could not convert value {raw_value!r} from column {self.value_column!r} "
                f"in CSV {self.csv_path!r} using {self.value_cast}."
            ) from e
