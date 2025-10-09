import csv
import json
from functools import cache
import sqlite3
import numpy as np


@cache
def get_timber_price_table(file_path: str) -> np.ndarray:
    """Converts the string representation of a timber price table csv to a numpy.ndarray."""
    table = np.genfromtxt(file_path, delimiter=';', skip_header=1)
    return table


@cache
def get_renewal_costs_as_dict(file_path: str) -> dict[str, float]:
    """Returns the csv at :file_path: as a dictionary, where key is the operation name and value ie the cost."""
    costs = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=';')
        _ = next(reader)  # skip header
        for row in reader:
            costs[row[0]] = float(row[1])  # operation: cost
    return costs


@cache
def get_land_values_as_dict(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_database_tables(db: sqlite3.Connection):
    cur = db.cursor()
    cur.execute(
        """
        CREATE TABLE nodes(identifier, stand, done_treatment, treatment_params, PRIMARY KEY(identifier, stand))
        """
    )
    cur.execute(
        """
        CREATE TABLE stands(node, identifier, year, management_unit_id, stand_id, area, area_weight, geo_location,
                            degree_days, owner_category, land_use_category, soil_peatland_category,
                            site_type_category, tax_class_reduction, tax_class, drainage_category,
                            drainage_feasibility, drainage_year, fertilization_year,
                            soil_surface_preparation_year, natural_regeneration_feasibility,
                            regeneration_area_cleaning_year, development_class, artificial_regeneration_year,
                            young_stand_tending_year, pruning_year, cutting_year, forestry_centre_id,
                            forest_management_category, method_of_last_cutting, municipality_id,
                            dominant_storey_age, area_weight_factors, fra_category,land_use_category_detail,
                            auxiliary_stand, sea_effect, lake_effect, basal_area,
                            PRIMARY KEY(node, identifier),
                            FOREIGN KEY(node, identifier) REFERENCES nodes(identifier, stand))
        """
    )
    cur.execute(
        """
        CREATE TABLE trees(node, stand, identifier, tree_number, species, breast_height_diameter, height,
                           measured_height, breast_height_age, biological_age, stems_per_ha, origin,
                           management_category, saw_log_volume_reduction_factor, pruning_year,
                           age_when_10cm_diameter_at_breast_height, stand_origin_relative_position,
                           lowest_living_branch_height, tree_category, storey, sapling, tree_type, tuhon_ilmiasu,
                           PRIMARY KEY (node, identifier),
                           FOREIGN KEY (node, stand) REFERENCES nodes(identifier, stand))
        """
    )
    cur.execute(
        """
        CREATE TABLE strata(node, stand, identifier, species, mean_diameter, mean_height, breast_height_age,
                            biological_age, stems_per_ha, basal_area, origin, management_category,
                            saw_log_volume_reduction_factor, cutting_year, age_when_10cm_diameter_at_breast_height,
                            tree_number, stand_origin_relative_position, lowest_living_branch_height, storey,
                            sapling_stems_per_ha, sapling_stratum, number_of_generated_trees,
                            PRIMARY KEY (node, identifier),
                            FOREIGN KEY (node, stand) REFERENCES nodes(identifier, stand))
        """
    )
