
from typing import Any
from enum import Enum
import numpy as np
from lukefi.metsi.domain.forestry_types import ForestStand

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
    "land_use_category_detail",
    "auxiliary_stand",
    "area_weight_factor_0",
    "area_weight_factor_1",
    "stand_id",
    "basal_area",
    "dominant_storey_age",
    "main_tree_species_dominant_storey",
    "region",
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


def stand_internal_values(stand: ForestStand) -> list[Any]:
    # Must match STAND_INTERNAL_COLUMNS order (stand_as_internal_row)
    geo = stand.geo_location
    return [
        stand.year,
        stand.area,
        stand.area_weight,
        geo[0] if geo is not None else None,
        geo[1] if geo is not None else None,
        geo[2] if geo is not None else None,
        geo[3] if geo is not None else None,
        stand.degree_days,
        stand.owner_category,
        stand.land_use_category,
        stand.soil_peatland_category,
        stand.site_type_category,
        stand.tax_class_reduction,
        stand.tax_class,
        stand.drainage_category,
        stand.drainage_year,
        stand.fertilization_year,
        stand.soil_surface_preparation_year,
        stand.regeneration_area_cleaning_year,
        stand.development_class,
        stand.artificial_regeneration_year,
        stand.young_stand_tending_year,
        stand.cutting_year,
        stand.forestry_centre_id,
        stand.forest_management_category,
        stand.method_of_last_cutting,
        stand.municipality_id,
        stand.fra_category,
        stand.land_use_category_detail,
        stand.auxiliary_stand,
        stand.area_weight_factors[0],
        stand.area_weight_factors[1],
        stand.stand_id,
        stand.basal_area,
        stand.dominant_storey_age,
        stand.main_tree_species_dominant_storey,
        stand.region,
    ]
