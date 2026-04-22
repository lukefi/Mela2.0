from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from lukefi.metsi.data.conversion import vmi2internal
from lukefi.metsi.data.formats import util, vmi_util
from lukefi.metsi.data.formats.declarative_conversion import ConversionMapper
from lukefi.metsi.data.model import ForestStand
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

    stand_rows: list[str]
    stratum_rows: list[str]
    tree_rows: list[str]

    builder_flags: dict[str, bool]
    conversion_reader: ConversionMapper

    def __init__(self, builder_flags: dict[str, bool], data_rows: list[str]) -> None:
        self.stand_rows = []
        self.stratum_rows = []
        self.tree_rows = []

        self.builder_flags = builder_flags
        self._classify_rows(data_rows)

    def _classify_rows(self, data_rows: list[str]):
        for row in data_rows:
            kind = self._classify_row(row)
            if kind == RowKind.STAND:
                self.stand_rows.append(row)
            elif kind == RowKind.STRATUM:
                self.stratum_rows.append(row)
            elif kind == RowKind.TREE:
                self.tree_rows.append(row)

    @abstractmethod
    def _classify_row(self, row: str) -> RowKind:
        pass

    def _convert_stand_entry(self, source_data: dict[str, str], stand_id: Optional[int]=None) -> ForestStand:
        result = ForestStand()

        result.identifier = self._generate_stand_identifier(source_data)
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

    def _generate_stand_identifier(self, source_data: dict[str, str]) -> str:
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
        super().__init__(builder_flags, data_rows)

    def _classify_row(self, row: str) -> RowKind:
        return RowKind(row[0])

    def _convert_stand_entry(self, source_data: dict[str, str], stand_id: Optional[int] = None) -> ForestStand:
        result = super()._convert_stand_entry(source_data, stand_id)

        result.year = vmi_util.parse_vmi13_date(source_data["date"]).year
        result.start_year = result.year

        area_ha = vmi_util.determine_vmi13_area_ha(
            int(source_data["county"]),
            int(source_data["lohkomuoto"]),
            util.get_or_default(
                util.parse_int(source_data["lohkotarkenne"])
            )
        )

        result.area_weight_factors = vmi_util.determine_area_factors(
            source_data["osuus4m"],
            source_data["lohkomuoto"]
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
