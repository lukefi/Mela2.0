from typing import Any
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.util.lookup_table import LookupTable


def dd_group_for(degree_days: int) -> int:
    # Example: real logic later; dummy now
    # 0–1200 -> 1, 1201–1400 -> 2, etc.
    if degree_days < 1200:
        return 1
    if degree_days < 1400:
        return 2
    return 3


def site_group_for(site_type_category: int | Any) -> int:
    v = getattr(site_type_category, "value", site_type_category)
    if v is None:
        return 1
    if 1 <= v <= 2:
        return 1
    if v == 3:
        return 2
    if v == 4:
        return 3
    if 5 <= v <= 8:
        return 4
    raise ValueError(f"Unsupported site_type_category={v!r}; expected 1..8.")


def species_group_for(_stand: ForestStand) -> int:
    return 1  # pine


def min_stems_table(csv_path: str = "min_stems.csv") -> LookupTable[ForestStand]:
    """
    Factory for the min stems lookup table.

    csv_path is kept as a parameter so user_events.py Events
    can override it via parameters["min_stems_csv"] if they want.
    """
    return LookupTable[ForestStand](
        csv_path=csv_path,
        key_columns=[
            "site_type_category",   # must exist on ForestStand
            "species",        # must exist on ForestStand
            "degree_days",          # must exist on ForestStand
        ],
        value_column="min_stems",
        transforms={
            "degree_days": dd_group_for,
            "site_type_category": site_group_for,
            "species": species_group_for,
        },
        value_cast=int,
    )
