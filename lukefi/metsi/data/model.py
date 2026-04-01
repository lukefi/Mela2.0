from copy import copy
from enum import Enum
import dataclasses
import sqlite3
from typing import Optional, override
from dataclasses import dataclass

import numpy as np
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.data.enums.internal import (
    CuttingMethod,
    DevelopmentClass,
    FraLandUseClass,
    LandUseCategory,
    OwnerCategory,
    SiteType,
    SoilPeatlandCategory,
    TreeManagementCategory,
    TreeSpecies,
    DrainageCategory,
    PeatlandForestType,
    DrainedPeatlandForestType)
from lukefi.metsi.data.formats.util import convert_str_to_type as conv
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.domain.utils.file_io import STANDS_TYPES, TREES_TYPES, STRATA_TYPES
from lukefi.metsi.forestry.volume import tree_volumes
from lukefi.metsi.sim.finalizable import Finalizable

# NOTE:
# * the deepcopy methods here are roughly equivalent to
#       def __deepcopy__(self, memo):
#           return cls(**self.__dict__)
#   but __new__ + update() is ~25% faster (tested on Python 3.10).
#   dict.copy() vs dict(other) vs dict.update(other) are all equally fast.
# * none of the ForestStand / ReferenceTree / TreeStratum have their __init__
#   methods run when copied. don't add a (non-trivial) __init__ method to any class here.
# * if you add any containers on any class here, you need to add a manual copy
#   in the __deepcopy__ method. see ForestStand.__deepcopy__ for an example.


@dataclass(init=True, repr=False, order=False, unsafe_hash=False, frozen=False, match_args=False, kw_only=False,
           slots=False, weakref_slot=False, eq=False)
class ForestStand(Finalizable, ComputationalUnit):
    # VMI data type 1
    # SMK data type Stand

    reference_trees: ReferenceTrees = dataclasses.field(default_factory=ReferenceTrees)
    tree_strata: TreeStrata = dataclasses.field(default_factory=TreeStrata)

    time: int = 0
    start_time: int = 0

    # unique identifier for entity within its domain
    identifier: str = ""

    stand_id: Optional[int] = None

    area: float = 0.0

    area_weight: float = area
    # lat, lon, height above sea level (m), CRS
    geo_location: Optional[tuple[Optional[float], Optional[float], Optional[float], Optional[str]]] = None

    degree_days: Optional[float] = None
    owner_category: Optional[OwnerCategory] = None
    land_use_category: Optional[LandUseCategory] = None
    soil_peatland_category: Optional[SoilPeatlandCategory] = None
    site_type_category: Optional[SiteType] = None
    tax_class_reduction: Optional[int] = None
    tax_class: Optional[int] = None
    drainage_category: Optional[DrainageCategory] = None
    drainage_year: Optional[int] = None
    fertilization_year: Optional[int] = None
    soil_surface_preparation_year: Optional[int] = None
    regeneration_area_cleaning_year: Optional[int] = None
    development_class: Optional[DevelopmentClass] = None
    main_tree_species_dominant_storey: Optional[TreeSpecies] = None
    artificial_regeneration_year: Optional[int] = None
    young_stand_tending_year: Optional[int] = None
    cutting_year: Optional[int] = None
    forestry_centre_id: Optional[int] = None
    forest_management_category: Optional[int | float] = None
    method_of_last_cutting: Optional[CuttingMethod] = None
    municipality_id: Optional[int] = None
    ds_main_tree_species_biological_age: Optional[float] = None
    ds_dominant_height: Optional[float] = None

    # stand specific factors for scaling estimated ReferenceTree count per hectare
    area_weight_factors: tuple[float, float] = (1.0, 1.0)

    fra_category: Optional[FraLandUseClass] = None  # VMI fra category
    # VMI stand number > 1 (meaning sivukoeala, auxiliary stand)
    auxiliary_stand: bool = False
    sea_effect: Optional[float] = None
    lake_effect: Optional[float] = None

    basal_area: Optional[float] = None
    stems_per_ha: Optional[float] = None
    ds_ba_weighted_mean_diameter: Optional[float] = None
    ds_ba_weighted_mean_height: Optional[float] = None
    region: Optional[int] = None
    ahvkeilaus: Optional[str] = None  # only used in VMI11

    peatland_type: Optional[PeatlandForestType] = None
    drained_peatland_type: Optional[DrainedPeatlandForestType] = None
    under_storey: bool = False
    over_storey: bool = False
    ds_main_tree_species: Optional[TreeSpecies] = None
    sqlite_decl: Optional[dict] = None

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return id(self)

    @property
    def year(self):
        return self.time

    @year.setter
    def year(self, value):
        self.time = value

    @property
    def start_year(self):
        return self.start_time

    @start_year.setter
    def start_year(self, value):
        self.start_time = value

    @property
    def relative_year(self):
        return self.relative_time

    def set_identifiers(self, stand_id: Optional[int]):
        self.stand_id = stand_id

    def set_area(self, area_ha: float | None):
        if area_ha is None:
            raise MetsiException("Area missing")
        if self.is_auxiliary():
            self.area = 0.0
        else:
            self.area = area_ha
        self.area_weight = area_ha

    def set_geo_location(self, lat: Optional[float], lon: Optional[float],
                         height: Optional[float], system: str = "EPSG:3067"):
        if not lat or not lon:
            raise ValueError("Invalid source values for geo location")
        self.geo_location = (lat, lon, height, system)

    def validate(self):
        pass

    def is_auxiliary(self):
        return self.auxiliary_stand

    def is_forest_land(self):
        return (self.land_use_category < LandUseCategory.OTHER_FOREST) if self.land_use_category is not None else False

    def has_trees(self):
        return len(self.reference_trees) > 0

    def has_strata(self):
        return len(self.tree_strata) > 0

    def from_row(self, row):
        self.year = conv(row[0], int)
        self.start_year = self.year
        self.area = conv(row[1], float) or 0.0
        self.area_weight = conv(row[2], float) or 0.0

        self.geo_location = (
            conv(row[3], float),
            conv(row[4], float),
            conv(row[5], float),
            conv(row[6], str)
        )
        self.degree_days = conv(row[7], float)
        self.owner_category = conv(row[8], OwnerCategory)
        self.land_use_category = conv(row[9], LandUseCategory)
        self.soil_peatland_category = conv(row[10], SoilPeatlandCategory)
        self.site_type_category = conv(row[11], SiteType)
        self.tax_class_reduction = conv(row[12], int)
        self.tax_class = conv(row[13], int)
        self.drainage_category = conv(row[14], DrainageCategory)

        self.drainage_year = conv(row[15], int)
        self.fertilization_year = conv(row[16], int)
        self.soil_surface_preparation_year = conv(row[17], int)
        self.regeneration_area_cleaning_year = conv(row[18], int)
        self.development_class = DevelopmentClass(int(row[19])) if row[19] != 'None' else None
        self.artificial_regeneration_year = conv(row[20], int)
        self.young_stand_tending_year = conv(row[21], int)
        self.cutting_year = conv(row[22], int)
        self.forestry_centre_id = conv(row[23], int)
        self.forest_management_category = conv(row[24], float)
        self.method_of_last_cutting = CuttingMethod(int(row[25])) if row[25] != 'None' else None
        self.municipality_id = conv(row[26], int)

        self.fra_category = conv(row[27], FraLandUseClass)
        self.auxiliary_stand = row[28] == "True"
        self.area_weight_factors = (conv(row[29], float) or 0.0, conv(row[30], float) or 0.0)
        self.stand_id = conv(row[31], int)
        self.basal_area = conv(row[32], float)
        self.ds_main_tree_species_biological_age = conv(row[33], float)
        self.main_tree_species_dominant_storey = conv(row[34], TreeSpecies)
        self.region = conv(row[35], int)

        self.peatland_type = PeatlandForestType(int(row[36])) if row[36] != "None" else None
        self.drained_peatland_type = DrainedPeatlandForestType(int(row[37])) if row[37] != "None" else None
        self.under_storey = row[38] == "True"
        self.over_storey = row[39] == "True"

    @staticmethod
    def _sql_value(v):
        if v is None:
            return None
        if isinstance(v, Enum):
            return v.value
        # numpy scalar -> python scalar
        if isinstance(v, (np.generic,)):
            return v.item()
        # tuples / arrays that is store as TEXT
        if isinstance(v, (tuple, list)):
            return str(tuple(v))
        # numpy array row [x,y,z] -> (x, y, z)
        if isinstance(v, np.ndarray):
            return str(tuple(v.tolist()))
        # for bool as bool
        return v

    @classmethod
    def _decl_cols(cls, table: str, default: list[str]) -> list[str]:
        decl = cls.sqlite_decl
        if not decl:
            return default
        return list(decl.get(table, default))

    @classmethod
    def set_sqlite_decl(cls, decl: Optional[dict]) -> None:
        """ User's db output list are stored here """
        cls.sqlite_decl = decl

    @classmethod
    def from_csv_row(cls, row) -> "ForestStand":
        stand = cls()
        stand.identifier = row[1]
        stand.from_row(row[2:])
        return stand

    def get_value_list(self, keys: Optional[list[str]] = None) -> list:
        """ Returns instance values as list based on keys.
            If keys are not present all attribute values are returned """
        ad = []
        if keys is not None:
            ad = [getattr(self, k) for k in keys]  # Needs to fail noisy
        return ad

    @override
    def finalize(self):
        retval = copy(self)
        retval.reference_trees = self.reference_trees.finalize()
        retval.tree_strata = self.tree_strata.finalize()
        return retval

    @override
    def output_to_db(self, db: sqlite3.Connection, node: str):
        cur = db.cursor()

        def insert(table: str, cols: list[str], values: list):
            cur.execute(
                f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
                tuple(values),
            )

        # ---- stands ----
        stand_cols = self._decl_cols("stands", list(STANDS_TYPES.keys()))
        insert(
            "stands",
            ["node", "identifier"] + stand_cols,
            [node, self.identifier] + [self._sql_value(getattr(self, c)) for c in stand_cols],
        )

        # ---- trees ----
        tree_cols = self._decl_cols("trees", list(TREES_TYPES.keys()))
        tree_insert_cols = ["node", "stand", "identifier"] + tree_cols
        trees = self.reference_trees
        for i in range(trees.size):
            row = [self._sql_value(getattr(trees, c)[i]) for c in tree_cols]
            insert("trees", tree_insert_cols, [node, self.identifier, trees.identifier[i]] + row)

        # ---- strata ----
        strata_cols = self._decl_cols("strata", list(STRATA_TYPES.keys()))
        strata_insert_cols = ["node", "stand", "identifier"] + strata_cols
        strata = self.tree_strata
        for i in range(strata.size):
            row = [self._sql_value(getattr(strata, c)[i]) for c in strata_cols]
            insert("strata", strata_insert_cols, [node, self.identifier, strata.identifier[i]] + row)

    def update_aggregates(self):
        trees = self.reference_trees
        strata = self.tree_strata

        # ReferenceTrees
        trees.basal_area = np.pi * (trees.breast_height_diameter / 200) ** 2
        trees.volume = tree_volumes(trees, self.degree_days or 0.0)

        # ForestStand
        self.stems_per_ha = np.sum(trees.stems_per_ha) + np.sum(strata.stems_per_ha)
        self.basal_area = np.sum(trees.stems_per_ha *
                                 trees.basal_area) + np.sum(strata.basal_area)
        self.ds_ba_weighted_mean_diameter = (
            (np.sum(
                trees.stems_per_ha *
                trees.basal_area *
                trees.breast_height_diameter) +
                np.sum(
                strata.basal_area *
                strata.mean_diameter)) /
            self.basal_area) if (
                self.basal_area > 0) else None

        self.ds_ba_weighted_mean_height = ((np.sum(trees.stems_per_ha * trees.basal_area * trees.height) +
                                            np.sum(strata.basal_area * strata.mean_height)) /
                                           self.basal_area) if (self.basal_area > 0) else None

        self.ds_dominant_height = self._calculate_dominant_height()

    def _calculate_dominant_height(self) -> float | None:
        if len(self.reference_trees) == 0:
            return None
        trees = self.reference_trees
        non_saved_trees_indices = np.flatnonzero(trees.management_category != TreeManagementCategory.RETENTION_TREE)
        sorted_trees_indices = np.flip(np.argsort(trees.breast_height_diameter[non_saved_trees_indices]))
        sorted_cum_stems = np.cumsum(trees.stems_per_ha[non_saved_trees_indices][sorted_trees_indices])
        i_100_largest_arr = np.flatnonzero(sorted_cum_stems >= 100)
        if len(i_100_largest_arr) == 0:
            stems_smallest: float = trees.stems_per_ha[non_saved_trees_indices][sorted_trees_indices][-1]
            i_100_largest: int = len(non_saved_trees_indices) - 1
        elif i_100_largest_arr[0] == 0:
            stems_smallest = 100.0
            i_100_largest = 0
        else:
            i_100_largest = i_100_largest_arr[0]
            stems_smallest = 100 - sorted_cum_stems[i_100_largest - 1]

        numerator_1 = np.sum(trees.stems_per_ha[non_saved_trees_indices][sorted_trees_indices][:i_100_largest] *
                             trees.height[non_saved_trees_indices][sorted_trees_indices][:i_100_largest])
        numerator_2: float = stems_smallest * trees.height[non_saved_trees_indices][sorted_trees_indices][i_100_largest]
        denominator = min(100, sorted_cum_stems[i_100_largest])

        return (numerator_1 + numerator_2) / denominator


def stand_as_internal_csv_row(stand: ForestStand, decl_keys: Optional[list[str]] = None) -> list[str]:
    result = ["stand", stand.identifier]
    result.extend(stand_as_internal_row(stand))
    if decl_keys is not None:
        result.extend(stand.get_value_list(decl_keys))
    return result


def stand_as_rst_row(stand: ForestStand):
    return [
        stand.stand_id,
        stand.year,
        stand.area,
        stand.area_weight,
        stand.geo_location[0] if stand.geo_location else None,
        stand.geo_location[1] if stand.geo_location else None,
        stand.stand_id,
        stand.geo_location[2] if stand.geo_location else None,
        stand.degree_days,
        stand.owner_category,
        stand.land_use_category,
        stand.soil_peatland_category,
        stand.site_type_category,
        stand.tax_class_reduction,
        stand.tax_class,
        stand.drainage_category,
        None,
        None,
        stand.drainage_year,
        stand.fertilization_year,
        stand.soil_surface_preparation_year,
        None,
        stand.regeneration_area_cleaning_year,
        stand.development_class,
        stand.artificial_regeneration_year,
        stand.young_stand_tending_year,
        None,
        stand.cutting_year,
        stand.forestry_centre_id,
        stand.forest_management_category,
        stand.method_of_last_cutting,
        stand.municipality_id,
        None,
        stand.ds_main_tree_species_biological_age,
    ]


def stand_as_internal_row(stand: ForestStand):
    return [
        stand.year,
        stand.area,
        stand.area_weight,
        stand.geo_location[0] if stand.geo_location is not None else None,
        stand.geo_location[1] if stand.geo_location is not None else None,
        stand.geo_location[2] if stand.geo_location is not None else None,
        stand.geo_location[3] if stand.geo_location is not None else None,
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
        stand.auxiliary_stand,
        stand.area_weight_factors[0],
        stand.area_weight_factors[1],
        stand.stand_id,
        stand.basal_area,
        stand.ds_main_tree_species_biological_age,
        stand.main_tree_species_dominant_storey,
        stand.region,
        stand.peatland_type,
        stand.drained_peatland_type,
        stand.under_storey,
        stand.over_storey,
    ]
