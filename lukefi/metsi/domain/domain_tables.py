from typing import Any
from pathlib import Path
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.util.lookup_table import LookupTable
from lukefi.metsi.data.enums.internal import SiteType, TreeSpecies


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
        return SiteType.VERY_RICH_SITE.value
    if SiteType.VERY_RICH_SITE.value <= v <= SiteType.RICH_SITE.value:
        return SiteType.VERY_RICH_SITE.value
    if v == SiteType.DAMP_SITE.value:
        return SiteType.RICH_SITE.value
    if v == SiteType.SUB_DRY_SITE.value:
        return SiteType.DAMP_SITE.value
    if SiteType.DRY_SITE.value <= v <= SiteType.OPEN_MOUNTAINS.value:
        return SiteType.SUB_DRY_SITE.value
    raise ValueError(f"Unsupported site_type_category={v!r}; expected 1..8.")


def species_group_for(_stand: ForestStand) -> int:
    return TreeSpecies.PINE.value


def min_stems_table(csv_path: str | Path = "") -> LookupTable[ForestStand]:
    """
    Factory for the min stems lookup table.

    csv_path is kept as a parameter so user_events.py Events
    can override it via parameters["min_stems_csv"] if they want.
    """
    if csv_path == "":
        csv_path = Path("data") / "parameter_files" / "min_stems.csv"
    else:
        csv_path = Path(csv_path)

    return LookupTable[ForestStand](
        csv_path=str(csv_path),
        key_columns=[
            "site_type_category",
            "development_class",
            "degree_days",
        ],
        value_column="min_stems",
        transforms={
            "degree_days": dd_group_for,
            "site_type_category": site_group_for,
            "development_class": species_group_for,
        },
        value_cast=int,
    )
