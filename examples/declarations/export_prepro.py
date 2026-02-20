from lukefi.metsi.domain.exp_ops import *

default = {}  # Empty dict declares a default output content

mela_decl = {
    # 'additional_variables': ['VAR0', 'VAR1', 'VAR2', 'VAR_RANDOM'],
    'operations': [prepare_rst_output, classify_values_to],
    'operation_params': {
        classify_values_to: [
            {'format': 'rst'}
        ]
    }
}


mela = {
    'rst': mela_decl,
}

mela_and_default_csv = {
    'rst': mela_decl,
    'csv': default,
}

default_csv = {
    'csv': default
}

sqlite_decl = {
    "stands": [
        "year", "management_unit_id", "stand_id", "area", "area_weight", "geo_location", "degree_days",
        "owner_category", "land_use_category", "soil_peatland_category", "site_type_category",
        "tax_class_reduction", "tax_class", "drainage_category", "drainage_feasibility", "drainage_year",
        "fertilization_year", "soil_surface_preparation_year", "natural_regeneration_feasibility",
        "regeneration_area_cleaning_year", "development_class", "artificial_regeneration_year",
        "young_stand_tending_year", "pruning_year", "cutting_year", "forestry_centre_id",
        "forest_management_category", "method_of_last_cutting", "municipality_id", "dominant_storey_age",
        "area_weight_factors", "fra_category", "land_use_category_detail", "auxiliary_stand", "sea_effect",
        "lake_effect", "basal_area", "main_tree_species_dominant_storey", "dominant_height_dominant_storey",
        "region"
    ],
    "trees": [
        "tree_number", "species", "breast_height_diameter", "height", "measured_height", "breast_height_age",
        "biological_age", "stems_per_ha", "origin", "management_category", "saw_log_volume_reduction_factor",
        "pruning_year", "age_when_10cm_diameter_at_breast_height", "stand_origin_relative_position",
        "lowest_living_branch_height", "tree_category", "storey", "sapling", "tree_type", "tuhon_ilmiasu",
        "basal_area", "volume"
    ],
    "strata": [
        "species", "mean_diameter", "mean_height", "breast_height_age", "biological_age", "stems_per_ha",
        "basal_area", "origin", "management_category", "saw_log_volume_reduction_factor", "cutting_year",
        "age_when_10cm_diameter_at_breast_height", "tree_number", "stand_origin_relative_position",
        "lowest_living_branch_height", "storey", "sapling_stems_per_ha", "sapling_stratum",
        "number_of_generated_trees"
    ],
}


__all__ = ['mela', 'default_csv', 'mela_and_default_csv',
           "sqlite_decl",]
