
from typing import Any
from enum import Enum
import numpy as np

STAND_INTERNAL_COLUMNS: list[str] = [
    "year",
    "area",
    "area_weight",
    "geo_lat",
    "geo_lon",
    "geo_height",
    "geo_crs",
    "degree_days",
    "owner_category",
    "land_use_category",
    "soil_peatland_category",
    "site_type_category",
    "tax_class_reduction",
    "tax_class",
    "drainage_category",
    "drainage_year",
    "fertilization_year",
    "soil_surface_preparation_year",
    "regeneration_area_cleaning_year",
    "development_class",
    "artificial_regeneration_year",
    "young_stand_tending_year",
    "cutting_year",
    "forestry_centre_id",
    "forest_management_category",
    "method_of_last_cutting",
    "municipality_id",
    "fra_category",
    "auxiliary_stand",
    "area_weight_factor_0",
    "area_weight_factor_1",
    "stand_id",
    "basal_area",
    "dominant_storey_age",
    "main_tree_species_dominant_storey",
    "region",
    "peatland_type",
    "drained_peatland_type",
    "under_storey",
    "over_storey",
]


def csv_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, Enum):
        return str(v.value)
    # numpy scalar → python scalar
    if isinstance(v, (np.generic,)):
        v = v.item()
    # normalize NaN
    try:
        if isinstance(v, float) and np.isnan(v):
            return ""
    except (ValueError, KeyError):
        pass
    return str(v)
