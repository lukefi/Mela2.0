import ast
from copy import copy
import dataclasses
from enum import Enum
from functools import lru_cache
import sqlite3
from typing import Any, Optional, override
from dataclasses import dataclass

import numpy as np
from lukefi.metsi.data.enums.internal import (
    CRS,
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
from lukefi.metsi.data.motti.motti_types import MottiState
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.forestry.volume import tree_volumes
from lukefi.metsi.core.exceptions import MetsiException
from lukefi.metsi.core.model import ComputationalUnit, Finalizable
from lukefi.metsi.core.treatment import PredeterminedTreatment
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL

STANDS_TYPES = {
    "year": "INTEGER",
    "stand_id": "INTEGER",
    "area": "REAL",
    "area_weight": "REAL",
    "geo_location": "TEXT",
    "degree_days": "REAL",
    "owner_category": "INTEGER",
    "land_use_category": "INTEGER",
    "soil_peatland_category": "INTEGER",
    "site_type_category": "INTEGER",
    "tax_class_reduction": "INTEGER",
    "tax_class": "INTEGER",
    "drainage_category": "INTEGER",
    "drainage_year": "INTEGER",
    "fertilization_year": "INTEGER",
    "soil_surface_preparation_year": "INTEGER",
    "regeneration_area_cleaning_year": "INTEGER",
    "development_class": "INTEGER",
    "artificial_regeneration_year": "INTEGER",
    "young_stand_tending_year": "INTEGER",
    "cutting_year": "INTEGER",
    "forestry_centre_id": "INTEGER",
    "forest_management_category": "REAL",
    "method_of_last_cutting": "INTEGER",
    "municipality_id": "INTEGER",
    "ds_main_tree_species_biological_age": "REAL",
    "area_weight_factors": "TEXT",
    "fra_category": "INTEGER",
    "auxiliary_stand": "INTEGER",
    "sea_effect": "REAL",
    "lake_effect": "REAL",
    "basal_area": "REAL",
    "main_tree_species_dominant_storey": "INTEGER",
    "ds_dominant_height": "REAL",
    "region": "INTEGER",
    "peatland_type": "INTEGER",
    "drained_peatland_type": "INTEGER",
    "under_storey": "INTEGER",
    "over_storey": "INTEGER",

}

TREES_TYPES = {
    "tree_number": "INTEGER",
    "species": "INTEGER",
    "breast_height_diameter": "REAL",
    "height": "REAL",
    "measured_height": "REAL",
    "breast_height_age": "REAL",
    "biological_age": "REAL",
    "stems_per_ha": "REAL",
    "origin": "INTEGER",
    "management_category": "INTEGER",
    "tree_category": "TEXT",
    "storey": "INTEGER",
    "sapling": "INTEGER",
    "tree_type": "TEXT",
    "damage_type": "TEXT",
    "basal_area": "REAL",
    "volume": "REAL",
    "stratum": "INTEGER"
}

STRATA_TYPES = {
    "species": "INTEGER",
    "mean_diameter": "REAL",
    "mean_height": "REAL",
    "breast_height_age": "REAL",
    "biological_age": "REAL",
    "stems_per_ha": "REAL",
    "basal_area": "REAL",
    "origin": "INTEGER",
    "stratum_number": "INTEGER",
    "storey": "INTEGER",
    "sapling_stems_per_ha": "REAL",
    "number_of_generated_trees": "INTEGER",
}


@dataclass(init=True, repr=False, order=False, unsafe_hash=False, frozen=False, match_args=False, kw_only=False,
           slots=True, weakref_slot=False, eq=False)
class ForestStand(Finalizable, ComputationalUnit):

    reference_trees: ReferenceTrees = dataclasses.field(default_factory=ReferenceTrees)
    """
    Reference trees in the stand.
    """
    tree_strata: TreeStrata = dataclasses.field(default_factory=TreeStrata)
    """
    Tree strata in the stand.
    """
    motti_state: Optional[MottiState] = None

    time: int = 0
    """
    Current time for the simulation unit [a].
    """
    start_time: int = 0
    """
    Starting time for the simulation unit [a].
    """

    identifier: str = ""
    """
    Unique free-form identifier for the forest stand.
    """
    stand_id: Optional[int] = None
    """
    Running unique identifier number for forest stands.
    """
    area: float = 0.0
    """
    Area of the stand [ha].
    """
    area_weight: float = area
    """
    Area weight for growing stock [ha].
    """
    geo_location: Optional[tuple[Optional[float], Optional[float], Optional[float], Optional[CRS]]] = None
    """
    Latitude, longitude, height above sea level [m] and coordinate reference system (CRS).
    """
    degree_days: Optional[float] = None
    """
    Temperature sum: sum of daily average temperatures over 4 °C, starting from 1 Jan. [°Cd].
    """
    owner_category: Optional[OwnerCategory] = None
    """
    Category of stand land owner.
    """
    land_use_category: Optional[LandUseCategory] = None
    """
    Land use category.
    """
    soil_peatland_category: Optional[SoilPeatlandCategory] = None
    """
    Soil peatland category.
    """
    site_type_category: Optional[SiteType] = None
    """
    Site type category.
    """
    tax_class_reduction: Optional[int] = None
    """
    Tax class reduction.
    """
    tax_class: Optional[int] = None
    """
    Tax class.
    """
    drainage_category: Optional[DrainageCategory] = None
    """
    Drainage category.
    """
    drainage_year: Optional[int] = None
    """
    Year of last drainage.
    """
    fertilization_year: Optional[int] = None
    """
    Year of last fertilization.
    """
    soil_surface_preparation_year: Optional[int] = None
    """
    Year of last soil surface preparation.
    """
    regeneration_area_cleaning_year: Optional[int] = None
    """
    Year of last regeneration area cleaning.
    """
    development_class: Optional[DevelopmentClass] = None
    """
    Development class of the forest stand.
    """
    main_tree_species_dominant_storey: Optional[TreeSpecies] = None
    """
    Main tree species in the dominant storey.
    """
    artificial_regeneration_year: Optional[int] = None
    """
    Year of the last artificial regeneration.
    """
    young_stand_tending_year: Optional[int] = None
    """
    Year of the last tending for a young stand.
    """
    cutting_year: Optional[int] = None
    """
    Year of the last performed cutting.
    """
    forestry_centre_id: Optional[int] = None
    """
    ID of the stand's forestry centre.
    """
    forest_management_category: Optional[int | float] = None
    """
    Forest management category.
    """
    method_of_last_cutting: Optional[CuttingMethod] = None
    """
    Method of the last performed cutting.
    """
    municipality_id: Optional[int] = None
    """
    ID of the municipality the stand is located in.
    """
    ds_main_tree_species_biological_age: Optional[float] = None
    """
    Current biological age of the main tree species in the dominant storey.
    """
    ds_dominant_height: Optional[float] = None
    """
    Dominant height in the dominant storey.
    """

    area_weight_factors: tuple[float, float] = (1.0, 1.0)
    """
    Proportions of the areas of the smaller and larger sample plot covered by the stand.
    """
    fra_category: Optional[FraLandUseClass] = None
    """
    NFI FRA class.
    """

    auxiliary_stand: bool = False
    """
    NFI stand number > 1 (meaning sivukoeala, auxiliary stand).
    """

    sea_effect: Optional[float] = None
    """
    Sea effect.
    """
    lake_effect: Optional[float] = None
    """
    Lake effect.
    """

    basal_area: Optional[float] = None
    """
    Basal area of the stand [m^2/ha].
    """
    stems_per_ha: Optional[float] = None
    """
    Number of stems per hectare in the stand [1/ha].
    """
    ds_ba_weighted_mean_diameter: Optional[float] = None
    """
    Mean diameter of dominant storey trees weighted by basal area.
    """
    ds_ba_weighted_mean_height: Optional[float] = None
    """
    Mean height of dominant storey trees weighted by basal area.
    """
    region: Optional[int] = None
    """
    Region where the stand is located.
    """
    ahvkeilaus: Optional[str] = None  # only used in VMI11
    """
    Scanning type for Ahvenanmaa.
    """

    peatland_type: Optional[PeatlandForestType] = None
    """
    Peatland type.
    """
    drained_peatland_type: Optional[DrainedPeatlandForestType] = None
    """
    Drained peatland type.
    """
    under_storey: bool = False
    """
    Whether the stand contains an under storey.
    """
    over_storey: bool = False
    """
    Whether the stand contains an over storey.
    """

    sqlite_decl: Optional[dict[str, list[str]]] = None
    """
    Declarations for SQLite output database columns.
    """

    predetermined_treatments: list[tuple[int, PredeterminedTreatment["ForestStand"]]] | None = None

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
                         height: Optional[float], system: CRS = CRS.EPSG_3067):
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
            conv(row[6], CRS)
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
        if isinstance(v, Enum):
            return repr(v.value)
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, (list, np.ndarray)):
            return str(v)
        if isinstance(v, tuple):
            return f'({", ".join(str(ForestStand._sql_value(x)) for x in v)})'
        return v

    @classmethod
    def _decl_cols(cls, table: str, default: list[str]) -> list[str]:
        decl = cls.sqlite_decl
        if not decl:
            return default
        return decl.get(table, default)

    @classmethod
    @lru_cache(maxsize=1)
    def _stands_cols(cls) -> list[str]:
        decl = cls.sqlite_decl
        if not decl:
            return list(STANDS_TYPES.keys())
        return decl.get("stands", list(STANDS_TYPES.keys()))

    @classmethod
    @lru_cache(maxsize=1)
    def _trees_cols(cls) -> list[str]:
        decl = cls.sqlite_decl
        if not decl:
            return list(TREES_TYPES.keys())
        return decl.get("trees", list(TREES_TYPES.keys()))

    @classmethod
    @lru_cache(maxsize=1)
    def _strata_cols(cls) -> list[str]:
        decl = cls.sqlite_decl
        if not decl:
            return list(STRATA_TYPES.keys())
        return decl.get("strata", list(STRATA_TYPES.keys()))

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

        if self.motti_state is not None:
            retval.motti_state = Motti4DLL.clone_state(self.motti_state)

        return retval

    @classmethod
    @lru_cache(maxsize=1)
    def stands_insert_statement(cls):
        cols = cls._stands_cols()
        return f"INSERT INTO stands ({', '.join(["node", "identifier"] + cols)})"\
            f"VALUES({', '.join(['?'] * (len(cols) + 2))})"

    @classmethod
    @lru_cache(maxsize=1)
    def trees_insert_statement(cls):
        cols = cls._trees_cols()
        return f"INSERT INTO trees ({', '.join(["node", "stand", "identifier"] + cols)})"\
            f"VALUES({', '.join(['?'] * (len(cols) + 3))})"

    @classmethod
    @lru_cache(maxsize=1)
    def strata_insert_statement(cls):
        cols = cls._strata_cols()
        return f"INSERT INTO strata ({', '.join(["node", "stand", "identifier"] + cols)})"\
            f"VALUES({', '.join(['?'] * (len(cols) + 3))})"

    @override
    def output_initial_state_to_db(self, db: sqlite3.Connection):
        cur = db.cursor()
        cur.execute(
            """--sql
                INSERT INTO initial_stands
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """,
            (
                self.identifier,
                self.year,
                self.stand_id,
                self.area,
                self.area_weight,
                str(
                    (
                        float(self.geo_location[0]) if self.geo_location[0] is not None else None,
                        float(self.geo_location[1]) if self.geo_location[1] is not None else None,
                        float(self.geo_location[2]) if self.geo_location[2] is not None else None,
                        self.geo_location[3]
                    )
                ) if self.geo_location is not None else None,
                self.degree_days,
                self.owner_category,
                self.land_use_category,
                self.soil_peatland_category,
                self.site_type_category,
                self.tax_class_reduction,
                self.tax_class,
                self.drainage_category,
                self.drainage_year,
                self.fertilization_year,
                self.soil_surface_preparation_year,
                self.regeneration_area_cleaning_year,
                self.development_class,
                self.artificial_regeneration_year,
                self.young_stand_tending_year,
                self.cutting_year,
                self.forestry_centre_id,
                self.forest_management_category,
                self.method_of_last_cutting,
                self.municipality_id,
                self.ds_main_tree_species_biological_age,
                str(self.area_weight_factors),
                self.fra_category,
                self.auxiliary_stand,
                self.sea_effect,
                self.lake_effect,
                self.basal_area,
                self.main_tree_species_dominant_storey,
                self.ds_dominant_height,
                self.region,
                self.peatland_type,
                self.drained_peatland_type,
                self.under_storey,
                self.over_storey
            )
        )

        trees = self.reference_trees
        cur.executemany(
            """--sql
                INSERT INTO initial_trees
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                );
            """,
            (
                (
                    trees.identifier[i],
                    self.identifier,
                    int(trees.tree_number[i]),
                    int(trees.species[i]),
                    trees.breast_height_diameter[i],
                    trees.height[i],
                    trees.measured_height[i],
                    trees.breast_height_age[i],
                    trees.biological_age[i],
                    trees.stems_per_ha[i],
                    int(trees.origin[i]),
                    int(trees.management_category[i]),
                    trees.tree_category[i],
                    int(trees.storey[i]),
                    int(trees.sapling[i]),
                    trees.tree_type[i],
                    trees.damage_type[i],
                    trees.crown_class[i],
                    trees.basal_area[i],
                    trees.volume[i],
                    int(trees.stratum[i])
                )
                for i in range(trees.size)
            )
        )

    @override
    def output_to_db(self, db: sqlite3.Connection, node: str):
        cur = db.cursor()

        # ---- stands ----
        cur.execute(
            self.stands_insert_statement(),
            [node, self.identifier] + [self._sql_value(getattr(self, c)) for c in self._stands_cols()]
        )

        # ---- trees ----
        trees = self.reference_trees
        tree_attrs = [trees.identifier] + [getattr(trees, c) for c in self._trees_cols()]
        cur.executemany(
            self.trees_insert_statement(),
            (
                (
                    [node, self.identifier] + [self._sql_value(tree_attr[i]) for tree_attr in tree_attrs]
                )
                for i in range(trees.size)
            )
        )

        # ---- strata ----
        strata = self.tree_strata
        stratum_attrs = [strata.identifier] + [getattr(strata, c) for c in self._strata_cols()]
        cur.executemany(
            self.strata_insert_statement(),
            (
                (
                    [node, self.identifier] + [self._sql_value(stratum_attr[i]) for stratum_attr in stratum_attrs]
                )
                for i in range(strata.size)
            )
        )

    @override
    def update_aggregates(self):
        trees = self.reference_trees

        # ReferenceTrees
        trees.basal_area = np.pi * (trees.breast_height_diameter / 200) ** 2
        trees.volume = tree_volumes(trees, self.degree_days or 0.0)

        # ForestStand
        self.stems_per_ha = np.sum(trees.stems_per_ha)
        self.basal_area = np.sum(trees.stems_per_ha * trees.basal_area)
        self.ds_ba_weighted_mean_diameter = (
            (np.sum(
                trees.stems_per_ha *
                trees.basal_area *
                trees.breast_height_diameter)) / self.basal_area) if (self.basal_area > 0) else None

        self.ds_ba_weighted_mean_height = ((np.sum(trees.stems_per_ha * trees.basal_area * trees.height)) /
                                           self.basal_area) if (self.basal_area > 0) else None

        self.ds_dominant_height = self._calculate_dominant_height()

    def _calculate_dominant_height(self) -> float | None:
        if len(self.reference_trees) == 0:
            return None
        trees = self.reference_trees

        # Use only non-retention trees by default
        trees_indices = np.flatnonzero(trees.management_category != TreeManagementCategory.RETENTION_TREE)
        if len(trees_indices) == 0:
            # Fallback to using all trees if all are retention trees
            trees_indices = np.arange(len(trees))

        sorted_trees_indices = np.flip(np.argsort(trees.breast_height_diameter[trees_indices]))
        sorted_cum_stems = np.cumsum(trees.stems_per_ha[trees_indices][sorted_trees_indices])
        i_100_largest_arr = np.flatnonzero(sorted_cum_stems >= 100)
        if len(i_100_largest_arr) == 0:
            stems_smallest: float = trees.stems_per_ha[trees_indices][sorted_trees_indices][-1]
            i_100_largest: int = len(trees_indices) - 1
        elif i_100_largest_arr[0] == 0:
            stems_smallest = 100.0
            i_100_largest = 0
        else:
            i_100_largest = i_100_largest_arr[0]
            stems_smallest = 100 - sorted_cum_stems[i_100_largest - 1]

        numerator_1 = np.sum(trees.stems_per_ha[trees_indices][sorted_trees_indices][:i_100_largest] *
                             trees.height[trees_indices][sorted_trees_indices][:i_100_largest])
        numerator_2: float = stems_smallest * trees.height[trees_indices][sorted_trees_indices][i_100_largest]
        denominator = min(100, sorted_cum_stems[i_100_largest])

        return (numerator_1 + numerator_2) / denominator

    @classmethod
    @override
    def reconstruct_initial_state(cls, identifier: str, db: sqlite3.Connection) -> "ForestStand":
        cur = db.cursor()
        cur.row_factory = sqlite3.Row
        cur.execute(
            """--sql
                SELECT * FROM initial_stands
                WHERE
                    identifier = ?;
            """,
            (
                identifier,
            )
        )
        stand_row = cur.fetchone()

        cur.row_factory = None
        cur.execute(
            """--sql
                SELECT COUNT(*) FROM initial_trees
                WHERE
                    stand = ?;
            """,
            (
                identifier,
            )
        )
        tree_count = cur.fetchone()[0]
        trees = ReferenceTrees()
        trees.size = tree_count

        trees.identifier = np.array(_fetch_initial_trees_col(identifier, "identifier", cur), dtype=np.dtype("U30"))
        trees.tree_number = np.array(_fetch_initial_trees_col(identifier, "tree_number", cur), dtype=np.int32)
        trees.species = np.array(_fetch_initial_trees_col(identifier, "species", cur), dtype=np.int32)
        trees.breast_height_diameter = np.array(_fetch_initial_trees_col(
            identifier, "breast_height_diameter", cur), dtype=np.float64)
        trees.height = np.array(_fetch_initial_trees_col(identifier, "height", cur), dtype=np.float64)
        trees.measured_height = np.array(_fetch_initial_trees_col(identifier, "measured_height", cur), dtype=np.float64)
        trees.breast_height_age = np.array(_fetch_initial_trees_col(
            identifier, "breast_height_age", cur), dtype=np.float64)
        trees.biological_age = np.array(_fetch_initial_trees_col(identifier, "biological_age", cur), dtype=np.float64)
        trees.stems_per_ha = np.array(_fetch_initial_trees_col(identifier, "stems_per_ha", cur), dtype=np.float64)
        trees.origin = np.array(_fetch_initial_trees_col(identifier, "origin", cur), dtype=np.int32)
        trees.management_category = np.array(_fetch_initial_trees_col(
            identifier, "management_category", cur), dtype=np.int32)
        trees.tree_category = np.array(_fetch_initial_trees_col(identifier, "tree_category", cur), dtype=np.dtype("U1"))
        trees.storey = np.array(_fetch_initial_trees_col(identifier, "storey", cur), dtype=np.int32)
        trees.sapling = np.array(_fetch_initial_trees_col(identifier, "sapling", cur), dtype=np.bool_)
        trees.tree_type = np.array(_fetch_initial_trees_col(identifier, "tree_type", cur), dtype=np.dtype("U1"))
        trees.damage_type = np.array(_fetch_initial_trees_col(identifier, "damage_type", cur), dtype=np.dtype("U2"))
        trees.crown_class = np.array(_fetch_initial_trees_col(identifier, "crown_class", cur), dtype=np.dtype("U1"))
        trees.basal_area = np.array(_fetch_initial_trees_col(identifier, "basal_area", cur), dtype=np.float64)
        trees.volume = np.array(_fetch_initial_trees_col(identifier, "volume", cur), dtype=np.float64)
        trees.stratum = np.array(_fetch_initial_trees_col(identifier, "stratum", cur), dtype=np.int32)

        assert len(trees.identifier) == trees.size
        assert len(trees.tree_number) == trees.size
        assert len(trees.species) == trees.size
        assert len(trees.breast_height_diameter) == trees.size
        assert len(trees.height) == trees.size
        assert len(trees.measured_height) == trees.size
        assert len(trees.breast_height_age) == trees.size
        assert len(trees.biological_age) == trees.size
        assert len(trees.stems_per_ha) == trees.size
        assert len(trees.origin) == trees.size
        assert len(trees.management_category) == trees.size
        assert len(trees.tree_category) == trees.size
        assert len(trees.storey) == trees.size
        assert len(trees.sapling) == trees.size
        assert len(trees.tree_type) == trees.size
        assert len(trees.damage_type) == trees.size
        assert len(trees.crown_class) == trees.size
        assert len(trees.basal_area) == trees.size
        assert len(trees.volume) == trees.size
        assert len(trees.stratum) == trees.size

        retval = ForestStand(
            reference_trees=trees,
            tree_strata=TreeStrata(),
            motti_state=None,
            time=stand_row["year"],
            start_time=stand_row["year"],
            identifier=stand_row["identifier"],
            stand_id=stand_row["stand_id"],
            area=stand_row["area"],
            area_weight=stand_row["area_weight"],
            geo_location=_parse_geo_location(stand_row["geo_location"]),
            degree_days=stand_row["degree_days"],
            owner_category=conv(stand_row["owner_category"], OwnerCategory),
            soil_peatland_category=conv(stand_row["soil_peatland_category"], SoilPeatlandCategory),
            site_type_category=conv(stand_row["site_type_category"], SiteType),
            tax_class_reduction=stand_row["tax_class_reduction"],
            tax_class=stand_row["tax_class"],
            drainage_category=conv(stand_row["drainage_category"], DrainageCategory),
            drainage_year=stand_row["drainage_year"],
            fertilization_year=stand_row["fertilization_year"],
            soil_surface_preparation_year=stand_row["soil_surface_preparation_year"],
            regeneration_area_cleaning_year=stand_row["regeneration_area_cleaning_year"],
            development_class=conv(stand_row["development_class"], DevelopmentClass),
            artificial_regeneration_year=stand_row["artificial_regeneration_year"],
            young_stand_tending_year=stand_row["young_stand_tending_year"],
            cutting_year=stand_row["cutting_year"],
            forestry_centre_id=stand_row["forestry_centre_id"],
            forest_management_category=stand_row["forest_management_category"],
            method_of_last_cutting=conv(stand_row["method_of_last_cutting"], CuttingMethod),
            municipality_id=stand_row["municipality_id"],
            ds_main_tree_species_biological_age=stand_row["ds_main_tree_species_biological_age"],
            area_weight_factors=stand_row["area_weight_factors"],
            fra_category=conv(stand_row["fra_category"], FraLandUseClass),
            auxiliary_stand=bool(stand_row["auxiliary_stand"]),
            sea_effect=stand_row["sea_effect"],
            lake_effect=stand_row["lake_effect"],
            basal_area=stand_row["basal_area"],
            main_tree_species_dominant_storey=conv(stand_row["main_tree_species_dominant_storey"], TreeSpecies),
            ds_dominant_height=stand_row["ds_dominant_height"],
            region=stand_row["region"],
            peatland_type=conv(stand_row["peatland_type"], PeatlandForestType),
            drained_peatland_type=conv(stand_row["drained_peatland_type"], DrainedPeatlandForestType),
            under_storey=bool(stand_row["under_storey"]),
            over_storey=bool(stand_row["over_storey"])
        )

        return retval

    @classmethod
    @override
    def create_database_tables(cls, db: sqlite3.Connection, sqlite_decl: dict[str, str] | None = None):
        cur = db.cursor()

        # initial_stands
        cur.execute(
            """--sql
                CREATE TABLE initial_stands(
                    identifier TEXT,
                    year INTEGER,
                    stand_id INTEGER,
                    area REAL,
                    area_weight REAL,
                    geo_location TEXT,
                    degree_days REAL,
                    owner_category INTEGER,
                    land_use_category INTEGER,
                    soil_peatland_category INTEGER,
                    site_type_category INTEGER,
                    tax_class_reduction INTEGER,
                    tax_class INTEGER,
                    drainage_category INTEGER,
                    drainage_year INTEGER,
                    fertilization_year INTEGER,
                    soil_surface_preparation_year INTEGER,
                    regeneration_area_cleaning_year INTEGER,
                    development_class INTEGER,
                    artificial_regeneration_year INTEGER,
                    young_stand_tending_year INTEGER,
                    cutting_year INTEGER,
                    forestry_centre_id INTEGER,
                    forest_management_category REAL,
                    method_of_last_cutting INTEGER,
                    municipality_id INTEGER,
                    ds_main_tree_species_biological_age REAL,
                    area_weight_factors TEXT,
                    fra_category INTEGER,
                    auxiliary_stand INTEGER,
                    sea_effect REAL,
                    lake_effect REAL,
                    basal_area REAL,
                    main_tree_species_dominant_storey INTEGER,
                    ds_dominant_height REAL,
                    region INTEGER,
                    peatland_type INTEGER,
                    drained_peatland_type INTEGER,
                    under_storey INTEGER,
                    over_storey INTEGER,
                    PRIMARY KEY(identifier)
                );
            """
        )

        # initial_trees
        cur.execute(
            """--sql
                CREATE TABLE initial_trees(
                    identifier TEXT,
                    stand TEXT,
                    tree_number INTEGER,
                    species INTEGER,
                    breast_height_diameter REAL,
                    height REAL,
                    measured_height REAL,
                    breast_height_age REAL,
                    biological_age REAL,
                    stems_per_ha REAL,
                    origin INTEGER,
                    management_category INTEGER,
                    tree_category TEXT,
                    storey INTEGER,
                    sapling INTEGER,
                    tree_type TEXT,
                    damage_type TEXT,
                    crown_class INTEGER,
                    basal_area REAL,
                    volume REAL,
                    stratum INTEGER,
                    PRIMARY KEY(identifier)
                );
            """
        )

        # stands: required id fields + declared fields
        stand_cols = _select_columns("stands", sqlite_decl)
        # required id cols for stands table:
        stand_prefix = ["node TEXT", "identifier TEXT"]

        stand_decl = [f"{c} {STANDS_TYPES[c]}" for c in stand_cols]
        cur.execute(
            f"""--sql
            CREATE TABLE stands(
                {", ".join(stand_prefix + stand_decl)},
                PRIMARY KEY(node, identifier),
                FOREIGN KEY(node, identifier) REFERENCES nodes(identifier, unit))
            """
        )

        # trees: required id cols + declared cols
        tree_cols = _select_columns("trees", sqlite_decl)
        tree_prefix = ["node TEXT", "stand TEXT", "identifier TEXT"]
        tree_decl = [f"{c} {TREES_TYPES[c]}" for c in tree_cols]
        cur.execute(
            f"""--sql
            CREATE TABLE trees(
                {", ".join(tree_prefix + tree_decl)},
                PRIMARY KEY (node, identifier),
                FOREIGN KEY (node, stand) REFERENCES nodes(identifier, unit))
            """
        )

        # strata: required id cols + declared cols
        strata_cols = _select_columns("strata", sqlite_decl)
        strata_prefix = ["node TEXT", "stand TEXT", "identifier TEXT"]
        strata_decl = [f"{c} {STRATA_TYPES[c]}" for c in strata_cols]
        cur.execute(
            f"""--sql
            CREATE TABLE strata(
                {", ".join(strata_prefix + strata_decl)},
                PRIMARY KEY (node, identifier),
                FOREIGN KEY (node, stand) REFERENCES nodes(identifier, unit))
            """
        )


def _select_columns(table: str, decl: Optional[dict]) -> list[str]:
    if not decl:
        # default = all fields
        if table == "stands":
            return list(STANDS_TYPES.keys())
        if table == "trees":
            return list(TREES_TYPES.keys())
        if table == "strata":
            return list(STRATA_TYPES.keys())
        return []
    return list(decl.get(table, []))


def _parse_geo_location(src: str) -> tuple[float | None, float | None, float | None, str | None] | None:
    return ast.literal_eval(src)


def _fetch_initial_trees_col(stand: str, col: str, cur: sqlite3.Cursor) -> list[Any]:
    cur.execute(
        f"""--sql
            SELECT {col} FROM initial_trees
            WHERE
                stand = ?;
        """,
        (
            stand,
        )
    )
    return [row[0] for row in cur.fetchall()]


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
