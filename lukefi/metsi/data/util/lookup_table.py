from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Mapping, Sequence, TypeVar
import pandas as pd

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

            raw = getattr(stand, col)
            if self.transforms and col in self.transforms:
                raw = self.transforms[col](raw)

            key_values[col] = raw

        row = self._find_matching_row(key_values)

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

    def _find_matching_row(self, key_values: Mapping[str, Any]) -> Dict[str, Any]:

        csv_p = Path(self.csv_path).resolve()
        df = pd.read_csv(csv_p, dtype=str)  # ensure consistent string comparison

        if df.empty:
            raise ValueError(f"Lookup CSV {csv_p} has no data rows.")

        # Ensure all key columns exist
        missing_cols = [col for col in key_values if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"CSV {csv_p} is missing required key column(s) {missing_cols!r}."
            )

        # Build query mask
        mask = pd.Series(True, index=df.index)
        for col, key_val in key_values.items():
            mask &= (df[col] == str(key_val))

        filtered = df[mask]

        if filtered.empty:
            raise ValueError(
                f"No matching row in CSV {csv_p} for keys: "
                + ", ".join(f"{k}={v!r}" for k, v in key_values.items())
            )

        if len(filtered) > 1:
            raise ValueError(
                f"Ambiguous rows in CSV {csv_p} for keys: "
                + ", ".join(f"{k}={v!r}" for k, v in key_values.items())
            )

        row = filtered.iloc[0].to_dict()
        return {str(k): v for k, v in row.items()}
