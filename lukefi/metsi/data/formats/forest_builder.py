from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.conversion import vmi2internal
from lukefi.metsi.data.enums.vmi import VmiIteration
from lukefi.metsi.data.formats import util, vmi_util
from lukefi.metsi.data.formats.declarative_conversion import ConversionMapper
from lukefi.metsi.data.formats.vmi_const import VMI13_STAND_INDICES, VMI13_STRATUM_INDICES, VMI13_TREE_INDICES
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.domain.forestry_types import StandList


class RowKind(Enum):
    STAND = "1"
    STRATUM = "2"
    TREE = "3"


class ForestBuilder(ABC):

    @abstractmethod
    def build(self) -> StandList:
        pass


class VMIBuilder(ForestBuilder):

    stand_rows: list[dict[str, str]]
    stratum_rows: dict[str, list[dict[str, str]]]
    tree_rows: dict[str, list[dict[str, str]]]

    builder_flags: dict[str, bool]
    conversion_reader: ConversionMapper

    def __init__(self, builder_flags: dict[str, bool]) -> None:
        self.stand_rows = []
        self.stratum_rows = {}
        self.tree_rows = {}

        self.builder_flags = builder_flags

    @staticmethod
    @abstractmethod
    def _classify_row(row: str) -> RowKind:
        pass

    @classmethod
    def _convert_stand_entry(cls, source_data: dict[str, str], stand_id: Optional[int] = None) -> ForestStand:
        result = ForestStand()

        result.identifier = cls._generate_stand_identifier(source_data)
        result.stand_id = stand_id

        result.degree_days = vmi_util.transform_vmi_degree_days(source_data["degree_days"])
        result.owner_category = vmi2internal.convert_owner(source_data["owner_group"])
        result.fra_category = vmi2internal.convert_fra_land_use_class(source_data["fra_class"])
        result.land_use_category = vmi2internal.convert_land_use_category(source_data["land_category"])
        result.site_type_category = vmi2internal.convert_site_type_category(source_data["kasvupaikkatunnus"])
        result.soil_peatland_category = vmi2internal.convert_soil_peatland_category(source_data["paatyyppi"])
        result.tax_class_reduction = vmi_util.determine_tax_class_reduction(source_data["tax_class_reduction"])
        result.tax_class = vmi_util.determine_tax_class(source_data["tax_class"])
        result.drainage_category = vmi2internal.convert_drainage_category(source_data["ojitus_tilanne"])
        result.forestry_centre_id = vmi_util.parse_forestry_centre(source_data["forestry_centre"])
        result.municipality_id = util.parse_int(vmi_util.vmi_codevalue(source_data["municipality"]))

        result.auxiliary_stand = source_data["stand_number"] != '1'

        return result

    @staticmethod
    def _generate_stand_identifier(source_data: dict[str, str]) -> str:
        return source_data["lohkomuoto"] + "-" + \
            source_data["section_y"] + "-" + \
            source_data["section_x"] + "-" + \
            source_data["test_area_number"] + "-" + \
            source_data["stand_number"]


class VMI9Builder(VMIBuilder):
    pass


class VMI10Builder(VMIBuilder):
    pass


class VMI11Builder(VMIBuilder):
    pass


class VMI12Builder(VMIBuilder):
    pass


class VMI13Builder(VMIBuilder):

    def __init__(self, builder_flags: dict[str, bool], data_rows: list[str]) -> None:
        super().__init__(builder_flags)
        for row in data_rows:
            kind = self._classify_row(row)
            split_row = row.split()
            if kind == RowKind.STAND:
                self.stand_rows.append({key: split_row[index] for key, index in VMI13_STAND_INDICES.items()})
            elif kind == RowKind.STRATUM and self.builder_flags.get("strata", False):
                stratum_row = {key: split_row[index] for key, index in VMI13_STRATUM_INDICES.items()}
                stand_identifier = vmi_util.generate_stand_identifier(stratum_row)
                self.stratum_rows.setdefault(stand_identifier, []).append(stratum_row)
            elif kind == RowKind.TREE and self.builder_flags.get("measured_trees", False):
                tree_row = {key: split_row[index] for key, index in VMI13_TREE_INDICES.items()}
                stand_identifier = vmi_util.generate_stand_identifier(tree_row)
                self.tree_rows.setdefault(stand_identifier, []).append(tree_row)

    @staticmethod
    def _classify_row(row: str) -> RowKind:
        return RowKind(row[0])

    def build(self) -> StandList:
        result: dict[str, ForestStand] = {}

        for i, row in enumerate(self.stand_rows):
            try:
                stand = self._convert_stand_entry(row, i + 1)
                result[stand.identifier] = stand
            except Exception as e:
                raise MetsiException(f"Parsing stand row {self.stand_rows[i]} failed: {e}") from e

        if self.builder_flags.get('strata', False):
            for stand_identifier, stand_stratum_rows in self.stratum_rows.items():
                strata = TreeStrata(len(stand_stratum_rows))
                for j, stand_stratum_row in enumerate(stand_stratum_rows):
                    self._convert_stratum_entry(strata, stand_stratum_row, j)

                result[stand_identifier].tree_strata = strata

        if self.builder_flags.get('measured_trees', False):
            for stand_identifier, stand_tree_rows in self.tree_rows.items():
                stand_ = result.get(stand_identifier)
                if stand_ is None:
                    continue

                trees = ReferenceTrees(len(stand_tree_rows))
                for j, stand_tree_row in enumerate(stand_tree_rows):
                    self._convert_tree_entry(trees, stand_tree_row, j, stand_.forestry_centre_id)

                result[stand_identifier].reference_trees = trees

        return list(result.values())

    @classmethod
    def _convert_stand_entry(cls, source_data: dict[str, str], stand_id: Optional[int] = None) -> ForestStand:
        result = super()._convert_stand_entry(source_data, stand_id)

        result.year = vmi_util.parse_vmi13_date(source_data["date"]).year
        result.start_year = result.year

        area_ha = vmi_util.determine_vmi13_area_ha(
            int(source_data["county"]),
            int(source_data["lohkomuoto"]),
            util.get_or_default(
                util.parse_int(source_data["lohkotarkenne"]),
                0
            )
        )

        result.area_weight_factors = vmi_util.determine_area_factors(
            source_data["osuus4m"],
            source_data["osuus9m"]
        )
        result.set_area(area_ha)

        lat = util.get_or_default(util.parse_type(source_data["lat_measured"], float), 0.0)
        lon = util.get_or_default(util.parse_type(source_data["lon_measured"], float), 0.0)
        if not lat:
            lat = util.get_or_default(util.parse_type(source_data["lat"], float), 0.0)
        if not lon:
            lon = util.get_or_default(util.parse_type(source_data["lon"], float), 0.0)

        height = vmi_util.transform_vmi13_height_above_sea_level(source_data["height_above_sea_level"])
        result.set_geo_location(lat, lon, height)

        result.drainage_year = vmi_util.determine_drainage_year(source_data["ojitus_aika"], result.year)
        result.fertilization_year = None
        result.soil_surface_preparation_year = vmi_util.determine_soil_surface_preparation_year(
            source_data["maanmuokkaus_aika"],
            result.year
        )
        result.regeneration_area_cleaning_year = vmi_util.determine_clearing_of_reform_sector_year(
            source_data["muu_toimenpide"],
            source_data["muu_toimenpide_aika"],
            result.year
        )
        result.artificial_regeneration_year = vmi_util.determine_artificial_regeneration_year(
            source_data["viljely"],
            source_data["viljely_aika"],
            result.year
        )
        maintenance_details = vmi2internal.convert_forest_maintenance_details(
            source_data["hakkuu_tapa"],
            source_data["hakkuu_aika"],
            result.year
        )
        result.young_stand_tending_year = maintenance_details[0]
        result.cutting_year = maintenance_details[1]
        result.method_of_last_cutting = maintenance_details[2]
        result.ds_main_tree_species_biological_age = vmi_util.determine_vmi13_dominant_storey_age(
            source_data["vallitsevanjaksonika"]
        )

        result.peatland_type = vmi2internal.convert_peatland_forest_type(source_data["suotyy"])
        result.drained_peatland_type = vmi2internal.convert_drained_peatland_forest_type(source_data["tkgtyy"])
        result.under_storey = bool(util.parse_type(source_data["alikehl"], int))
        result.over_storey = bool(util.parse_type(source_data["ylikehl"], int))

        result.development_class = vmi2internal.convert_development_class(source_data["kehitysluokka"])
        result.main_tree_species_dominant_storey = vmi_util.determine_main_tree_species_dominant_storey(
            source_data["main_tree_species_dominant_storey"],
            result.site_type_category
        )
        result.basal_area = util.parse_type(source_data["pohjapintaala"], float)
        result.region = util.parse_int(source_data["county"])

        result.municipality_id = vmi_util.determine_municipality(
            source_data["municipality"],
            source_data["kitukunta"]
        )

        if result.land_use_category and result.region is not None and result.owner_category is not None:
            is_ahvenanmaa = result.region == 21
            result.forest_management_category = vmi_util.determine_forest_management_category(
                result.land_use_category,
                is_ahvenanmaa,
                result.owner_category,
                source_data["puuntuotannon_rajoitus"],
                source_data["puuntuotannon_rajoitus_tarkenne"],
                source_data["muut_arvot"],
                source_data["suojametsakoodi"],
                source_data["ahvenanmaan_markkinahakkuualue"],
                source_data["koealan_kasittelyluokka"]
            )
        else:
            result.forest_management_category = 1

        return result

    @staticmethod
    def _convert_stratum_entry(strata: TreeStrata, row: dict[str, str], i: int):
        identifier = vmi_util.generate_stratum_identifier(row)

        species = vmi2internal.convert_species(row["species"])
        origin = vmi2internal.convert_origin(row["origin"])

        stems_per_ha = util.get_or_default(util.parse_type(row["stems_per_ha"], float), 0.0)
        sapling_stems_per_ha = util.get_or_default(util.parse_type(row["sapling_stems_per_ha"], float), 0.0)

        mean_diameter = util.parse_type(row["avg_diameter"], float)
        mean_height = vmi_util.determine_stratum_tree_height(row["avg_height"])

        biological_age, breast_height_age = vmi_util.determine_stratum_age_values(
            row["biological_age"],
            row["d13_age"],
            mean_height
        )

        basal_area = util.parse_type(row["basal_area"], float)
        stratum_number = util.parse_int(row["stratum_number"])
        storey = vmi_util.determine_storey_for_stratum(row["stratum_rank"])
        stratum_rank = vmi2internal.convert_stratum_rank(row["stratum_rank"])

        strata.identifier[i] = identifier
        strata.species[i] = species
        strata.mean_diameter[i] = mean_diameter
        strata.mean_height[i] = mean_height
        strata.breast_height_age[i] = breast_height_age
        strata.biological_age[i] = biological_age
        strata.stems_per_ha[i] = stems_per_ha
        strata.basal_area[i] = basal_area
        if origin is not None:
            strata.origin[i] = origin
        if stratum_number is not None:
            strata.stratum_number[i] = stratum_number
        if storey is not None:
            strata.storey[i] = storey
        strata.sapling_stems_per_ha[i] = sapling_stems_per_ha
        if stratum_rank is not None:
            strata.stratum_rank[i] = stratum_rank

    @staticmethod
    def _convert_tree_entry(trees: ReferenceTrees,
                            row: dict[str, str],
                            i: int,
                            forestry_centre_id: int | None,
                            ahvkeilaus: str | None = None,
                            height_conversion_factor: float = 100.0,
                            measured_height_conversion_factor: float = 10.0):
        identifier = vmi_util.generate_tree_identifier(row)
        tree_number = util.parse_type(row["tree_number"], int)
        species = vmi2internal.convert_species(row["species"])
        tree_category = vmi2internal.convert_tree_category(row["tree_category"])

        breast_height_diameter = vmi_util.transform_tree_diameter(row["diameter"])
        breast_height_age, biological_age = vmi_util.determine_tree_age_values(
            row["d13_age"],
            row["age_increase"],
            row["total_age"],
        )

        height = vmi_util.determine_tree_height(
            row["height"],
            conversion_factor=height_conversion_factor,
        )
        measured_height = vmi_util.determine_tree_height(
            row["measured_height"],
            conversion_factor=measured_height_conversion_factor,
        )

        stems_per_ha = vmi_util.determine_stems_per_ha(breast_height_diameter,
                                                       vmi_version=VmiIteration.VMI13,
                                                       forestry_centre_id=forestry_centre_id,
                                                       ahvkeilaus=ahvkeilaus)
        origin = vmi2internal.convert_origin(row["origin"])

        management_category = vmi_util.determine_tree_management_category(row["latvuskerros"])
        storey = vmi_util.determine_storey_for_tree(row["latvuskerros"])

        sapling = False
        tree_type = vmi2internal.convert_tree_type(row["tree_type"])

        damage_raw = row["tuhon_ilmiasu"]
        damage_type = None if damage_raw in ("  ", " ", ".", "") else damage_raw.strip()
        crown_class = vmi2internal.convert_crown_class(row["latvuskerros"])

        trees.identifier[i] = identifier
        if tree_number is not None:
            trees.tree_number[i] = tree_number
        trees.species[i] = species
        trees.breast_height_diameter[i] = breast_height_diameter
        trees.height[i] = height
        trees.measured_height[i] = measured_height
        trees.breast_height_age[i] = breast_height_age
        trees.biological_age[i] = biological_age
        trees.stems_per_ha[i] = stems_per_ha
        if origin is not None:
            trees.origin[i] = origin
        trees.management_category[i] = management_category
        trees.tree_category[i] = tree_category
        if storey is not None:
            trees.storey[i] = storey
        trees.sapling[i] = sapling
        trees.tree_type[i] = tree_type
        trees.damage_type[i] = damage_type
        trees.crown_class[i] = crown_class


class ForestCentreBuilder(ForestBuilder):
    ''' Base class for building a forest data model from Forest Centre (Suomen Metsakeskus) source '''

    @abstractmethod
    def build(self) -> StandList:
        ...

    @abstractmethod
    def convert_stand_entry(self, entry) -> ForestStand:
        ...

class XMLBuilder(ForestCentreBuilder):
    pass

class GeoPackageBuilder(ForestCentreBuilder):
    pass
