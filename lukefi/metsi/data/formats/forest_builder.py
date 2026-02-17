from collections.abc import Sequence, Iterable
from abc import ABC, abstractmethod
from typing import overload
import xml.etree.ElementTree as ET
from pandas import DataFrame, Series
import numpy as np

from lukefi.metsi.app.console_logging import print_logline
from lukefi.metsi.data.enums.internal import OwnerCategory
from lukefi.metsi.data.formats.vmi_const import (
    VMI12_STAND_INDICES,
    VMI12_STRATUM_INDICES,
    VMI12_TREE_INDICES,
    VMI13_STAND_INDICES,
    VMI13_STRATUM_INDICES,
    VMI13_TREE_INDICES
)
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.conversion import vmi2internal, fc2internal
from lukefi.metsi.data.formats import smk_util, util, vmi_util, gpkg_util
from lukefi.metsi.data.formats.declarative_conversion import ConversionMapper
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.domain.forestry_types import StandList
from lukefi.metsi.data.vector_model import DTYPES_TREE, DTYPES_STRATA


def _append_stratum_row(
    attr: dict[str, list],
    indices,
    row,
):
    """Append one VMI stratum row into an SoA attribute dict compatible with DTYPES_STRATA.

    No ConversionMapper / declared conversions are applied here.
    """

    identifier = vmi_util.generate_stratum_identifier(row, indices)
    species = vmi2internal.convert_species(row[indices["species"]])
    origin = vmi_util.determine_stratum_origin(row[indices["origin"]])

    stems_per_ha = util.get_or_default(util.parse_type(row[indices["stems_per_ha"]], float), 0.0)
    sapling_stems_per_ha = util.get_or_default(
        util.parse_type(row[indices["sapling_stems_per_ha"]], float), 0.0
    )

    mean_diameter = util.parse_type(row[indices["avg_diameter"]], float)
    mean_height = vmi_util.determine_stratum_tree_height(row[indices["avg_height"]])

    biological_age, breast_height_age = vmi_util.determine_stratum_age_values(
        row[indices["biological_age"]],
        row[indices["d13_age"]],
        mean_height,
    )

    basal_area = util.parse_type(row[indices["basal_area"]], float)
    tree_number = util.parse_int(row[indices["stratum_number"]])
    storey = vmi_util.determine_storey_for_stratum(row[indices["stratum_rank"]])

    # Defaults / placeholders (match DTYPES_STRATA fields)
    number_of_generated_trees = None

    values = {
        "identifier": identifier,
        "species": species,
        "mean_diameter": mean_diameter,
        "mean_height": mean_height,
        "breast_height_age": breast_height_age,
        "biological_age": biological_age,
        "stems_per_ha": stems_per_ha,
        "basal_area": basal_area,
        "origin": origin,
        "tree_number": tree_number,
        "storey": storey,
        "sapling_stems_per_ha": sapling_stems_per_ha,
        "number_of_generated_trees": number_of_generated_trees,
    }

    # Always append in DTYPES_STRATA order
    for key in DTYPES_STRATA:
        attr.setdefault(key, []).append(values.get(key, None))


def _append_tree_row(
    attr: dict[str, list],
    indices,
    row,
    is_vmi12: bool = False,
):
    """Append one VMI tree row into an SoA attribute dict compatible with DTYPES_TREE.

    No ConversionMapper / declared conversions are applied here.
    """

    identifier = vmi_util.generate_tree_identifier(row, indices)
    tree_number = util.parse_type(row[indices["tree_number"]], int)

    species = vmi2internal.convert_species(row[indices["species"]])
    tree_category = row[indices["tree_category"]]

    breast_height_diameter = vmi_util.transform_tree_diameter(row[indices["diameter"]])

    breast_height_age, biological_age = vmi_util.determine_tree_age_values(
        row[indices["d13_age"]],
        row[indices["age_increase"]],
        row[indices["total_age"]],
    )

    height = vmi_util.determine_tree_height(row[indices["height"]], conversion_factor=100.0)
    measured_height = vmi_util.determine_tree_height(row[indices["measured_height"]], conversion_factor=10.0)

    stems_per_ha = vmi_util.determine_stems_per_ha(breast_height_diameter, is_vmi12)

    origin = 0
    management_category = vmi_util.determine_tree_management_category(row[indices["latvuskerros"]])
    storey = vmi_util.determine_storey_for_tree(row[indices["latvuskerros"]])

    sapling = False
    tree_type = vmi_util.determine_tree_type(row[indices["tree_type"]])

    tuhon_raw = row[indices["tuhon_ilmiasu"]]
    tuhon_ilmiasu = None if tuhon_raw in ("  ", " ", ".", "") else tuhon_raw.strip()

    basal_area = None
    volume = None

    values = {
        "identifier": identifier,
        "tree_number": tree_number,
        "species": species,
        "breast_height_diameter": breast_height_diameter,
        "height": height,
        "measured_height": measured_height,
        "breast_height_age": breast_height_age,
        "biological_age": biological_age,
        "stems_per_ha": stems_per_ha,
        "origin": origin,
        "management_category": management_category,
        "tree_category": tree_category,
        "storey": storey,
        "sapling": sapling,
        "tree_type": tree_type,
        "tuhon_ilmiasu": tuhon_ilmiasu,
        "basal_area": basal_area,
        "volume": volume,
    }

    for key in DTYPES_TREE:
        attr.setdefault(key, []).append(values.get(key, None))


def _append_fc_stratum_row(attr: dict[str, list], stand_identifier: str, estratum: ET.Element):
    """
    Append one Forest Centre (XML) stratum row into an SoA attribute dict.
    """

    sd = smk_util.parse_stratum_data(estratum)

    tree_number = util.parse_type(sd.StratumNumber, int)
    raw_id = util.parse_type(sd.id, str)
    identifier = f"{stand_identifier}.{tree_number or raw_id}-stratum"

    basal_area = util.parse_type(sd.BasalArea, float)

    values = {
        "identifier": identifier,
        "species": fc2internal.convert_species(sd.TreeSpecies),
        "stems_per_ha": util.parse_type(sd.StemCount, float),
        "mean_diameter": util.parse_type(sd.MeanDiameter, float),
        "mean_height": util.parse_type(sd.MeanHeight, float),
        "breast_height_age": None,
        "biological_age": util.parse_type(sd.Age, float),
        "basal_area": basal_area,
        "origin": 0,
        "tree_number": tree_number,
        "storey": fc2internal.convert_storey(sd.Storey),
        "sapling_stems_per_ha": 0.0,
        "number_of_generated_trees": None,
    }

    for key in DTYPES_STRATA:
        attr.setdefault(key, []).append(values.get(key, None))


def _append_gpkg_stratum_row(attr: dict[str, list], stand_identifier: str, rowj: Series):
    """Append one GeoPackage stratum row into an SoA attribute dict.
    """

    tree_number = util.parse_type(rowj.stratumnumber, int)
    raw_id = util.parse_type(rowj.treestratumid, str)
    identifier = f"{stand_identifier}.{tree_number or raw_id}-stratum"

    basal_area = util.parse_type(rowj.basalarea, float)

    values = {
        "identifier": identifier,
        "species": fc2internal.convert_species(util.parse_type(rowj.treespecies, int, str)),
        "stems_per_ha": util.parse_type(rowj.stemcount, float),
        "mean_diameter": util.parse_type(rowj.meandiameter, float),
        "mean_height": util.parse_type(rowj.meanheight, float),
        "breast_height_age": None,
        "biological_age": util.parse_type(rowj.age, float),
        "basal_area": basal_area,
        "origin": None,
        "tree_number": tree_number,
        "storey": util.parse_type(rowj.storey, int),
        "sapling_stems_per_ha": 0.0,
        "number_of_generated_trees": None,
    }

    for key in DTYPES_STRATA:
        attr.setdefault(key, []).append(values.get(key, None))


class ForestBuilder(ABC):
    """Abstract base class of forest builders"""

    @abstractmethod
    def build(self) -> StandList:
        ...


class VMIBuilder(ForestBuilder):
    """Shared functionality of VMI* builders"""

    def __init__(self, builder_flags: dict, declared_conversions: dict, data_rows: Iterable):
        """
        Initialize instance variable lists for forest stands, reference trees and tree strata.
        Given source data is pre-parsed for data types 1, 2 and 3.

        :param builder_flags: building process spesific flags
        :param declared_conversions: callable data extractor
        :param data_rows: Iterable raw data rows from a VMI source file
        """
        self.forest_stands: list[str] = []
        self.reference_trees: list[str] = []
        self.tree_strata: list[str] = []
        self.builder_flags = builder_flags
        self.conversion_reader = ConversionMapper(declared_conversions)

        for row in data_rows:
            try:
                row_type = self.find_row_type(row)
                if row_type == 1:
                    self.forest_stands.append(row)
                elif row_type == 2:
                    self.tree_strata.append(row)
                elif row_type == 3:
                    self.reference_trees.append(row)
            except (IndexError, TypeError) as e:
                print(e)
                print('warning: VMI row not addressable: ')
                print('    ' + str(row))

    @overload
    def convert_stand_entry(self, indices: dict[str, int],
                            data_row: Sequence[str], stand_id: int | None = None) -> ForestStand:
        ...

    @overload
    def convert_stand_entry(self, indices: dict[str, slice],
                            data_row: str, stand_id: int | None = None) -> ForestStand:
        ...

    def convert_stand_entry(self, indices, data_row, stand_id=None) -> ForestStand:
        """Create a ForestStand out of given VMI type 1 data row using given data indices and order number"""
        result = ForestStand()
        result.identifier = vmi_util.generate_stand_identifier(data_row, indices)
        result.set_identifiers(stand_id)
        result.degree_days = vmi_util.transform_vmi_degree_days(data_row[indices["degree_days"]])
        result.owner_category = vmi2internal.convert_owner(data_row[indices["owner_group"]])
        result.fra_category = None if data_row[indices["fra_class"]] == '.' else data_row[indices["fra_class"]]
        result.land_use_category = vmi2internal.convert_land_use_category(data_row[indices["land_category"]])
        result.land_use_category_detail = data_row[indices["land_category_detail"]]
        result.site_type_category = vmi2internal.convert_site_type_category(data_row[indices["kasvupaikkatunnus"]])
        result.soil_peatland_category = vmi2internal.convert_soil_peatland_category(data_row[indices["paatyyppi"]])
        result.tax_class_reduction = vmi_util.determine_tax_class_reduction(data_row[indices["tax_class_reduction"]])
        result.tax_class = vmi_util.determine_tax_class(data_row[indices["tax_class"]])
        result.drainage_category = vmi2internal.convert_drainage_category(data_row[indices["ojitus_tilanne"]])
        result.development_class = vmi_util.determine_development_class(data_row[indices["kehitysluokka"]])
        result.main_tree_species_dominant_storey = vmi_util.determine_main_tree_species_dominant_storey(
            data_row[indices["main_tree_species_dominant_storey"]], result.site_type_category)
        result.forestry_centre_id = vmi_util.parse_forestry_centre(data_row[indices["forestry_centre"]])
        result.forest_management_category = vmi_util.determine_forest_management_category(
            result.land_use_category,
            result.forestry_centre_id,
            data_row,
            result.owner_category,
            indices, False)
        result.municipality_id = vmi_util.determine_municipality(
            data_row[indices["municipality"]],
            data_row[indices["kitukunta"]])
        result.auxiliary_stand = data_row[indices["stand_number"]] != '1'
        result.basal_area = util.parse_type(data_row[indices["pohjapintaala"]], float)
        result.region = util.parse_int(data_row[indices["county"]])
        return result

    @abstractmethod
    def find_row_type(self, row: str) -> int:
        ...

    @abstractmethod
    def build(self) -> StandList:
        ...


class VMI12Builder(VMIBuilder):
    """VMI12 specific builder implementation"""

    def __init__(self,
                 builder_flags: dict,
                 declared_conversions: dict,
                 data_rows: list[str] | None = None):
        if data_rows is None:
            data_rows = []
        super().__init__(builder_flags, declared_conversions, data_rows)

    def convert_stand_entry(self, indices, data_row, stand_id: int | None = None) -> ForestStand:
        """
        Create a ForestStand out of given VMI12 type 1 data row using given data indices and order number.
        Conversion variables are based fixed source indices.
        """
        # Fixed conversions
        result = super().convert_stand_entry(indices, data_row, stand_id)
        result.year = vmi_util.parse_vmi12_date(data_row[indices["date"]]).year
        result.start_year = result.year
        area_ha = vmi_util.determine_vmi12_area_ha(
            int(data_row[indices["lohkomuoto"]]),
            int(data_row[indices["county"]]))
        result.area_weight_factors = vmi_util.determine_area_factors(
            data_row[indices["osuus5m"]],
            data_row[indices["osuus9m"]]
        )
        result.set_area(area_ha)
        lat = util.parse_type(data_row[indices["lat"]], float)
        lon = util.parse_type(data_row[indices["lon"]], float)
        height = vmi_util.transform_vmi12_height_above_sea_level(data_row[indices["height_above_sea_level"]])
        result.set_geo_location(lat, lon, height, "EPSG:2393")
        result.drainage_year = vmi_util.determine_drainage_year(data_row[indices["ojitus_aika"]], result.year)
        result.soil_surface_preparation_year = vmi_util.determine_soil_surface_preparation_year(
            data_row[indices["maanmuokkaus"]],
            result.year)
        result.regeneration_area_cleaning_year = vmi_util.determine_clearing_of_reform_sector_year(
            data_row[indices["muu_toimenpide"]],
            data_row[indices["muu_toimenpide_aika"]],
            result.year)
        result.artificial_regeneration_year = vmi_util.determine_artificial_regeneration_year(
            data_row[indices["viljely"]],
            data_row[indices["viljely_aika"]],
            result.year)
        maintenance_details = vmi_util.determine_forest_maintenance_details(
            data_row[indices["hakkuu_tapa"]],
            data_row[indices["hakkuu_aika"]],
            result.year
        )
        result.young_stand_tending_year = maintenance_details[0]
        result.cutting_year = maintenance_details[1]
        result.method_of_last_cutting = maintenance_details[2]
        result.dominant_storey_age = vmi_util.determine_vmi12_dominant_storey_age(
            data_row[indices["vallitsevanjakson_d13ika"]],
            data_row[indices["vallitsevanjakson_ikalisays"]]
        )
        # Declared conversions
        result = self.conversion_reader.apply_conversions(result, data_row)
        return result

    def find_row_type(self, row: str) -> int:
        """Return VMI12 data type of the row"""
        return int(row[13])

    def build(self) -> StandList:
        """ Populate a list of ForestStand with associated ReferenceTrees and TreeStrata in SoA form
        """
        result: dict[str, ForestStand] = {}
        # Per-stand attribute dicts
        strata_attrs: dict[str, dict[str, list]] = {}
        tree_attrs: dict[str, dict[str, list]] = {}

        # Build stands
        for i, row in enumerate(self.forest_stands):
            stand = self.convert_stand_entry(VMI12_STAND_INDICES, row, i + 1)
            result[stand.identifier] = stand

        # Strata → TreeStrata SoA
        if self.builder_flags.get('strata', False):

            for row in self.tree_strata:
                stand_identifier = vmi_util.generate_stand_identifier(row, VMI12_STRATUM_INDICES)
                attr_dict = strata_attrs.setdefault(stand_identifier, {})
                _append_stratum_row(attr_dict, VMI12_STRATUM_INDICES, row)

        # Trees → ReferenceTrees SoA
        if self.builder_flags.get('measured_trees', False):
            for row in self.reference_trees:
                stand_identifier = vmi_util.generate_stand_identifier(row, VMI12_TREE_INDICES)
                attr_dict = tree_attrs.setdefault(stand_identifier, {})
                _append_tree_row(attr_dict, VMI12_TREE_INDICES, row, is_vmi12=True)

        # Attach SoA containers to stands
        for stand_id, stand in result.items():
            stand.tree_strata = TreeStrata().vectorize(strata_attrs.get(stand_id, {}))
            stand.reference_trees = ReferenceTrees().vectorize(tree_attrs.get(stand_id, {}))

        return list(result.values())


class VMI13Builder(VMIBuilder):
    """VMI13 specific builder implementation"""

    def __init__(self,
                 builder_flags: dict,
                 declared_conversions: dict,
                 data_rows: list[str] | None = None):
        if data_rows is None:
            data_rows = []
        pre_parsed_rows = map(lambda raw: raw.split(), data_rows)
        super().__init__(builder_flags, declared_conversions, pre_parsed_rows)

    def find_row_type(self, row: str) -> int:
        """Return VMI13 data type of the row"""
        return int(row[0])

    def convert_stand_entry(self, indices, data_row, stand_id: int | None = None) -> ForestStand:
        """Create a ForestStand out of given VMI13 type 1 data row using given data indices and order number"""
        # Fixed conversions
        result = super().convert_stand_entry(indices, data_row, stand_id)
        result.year = vmi_util.parse_vmi13_date(data_row[indices["date"]]).year
        result.start_year = result.year
        area_ha = vmi_util.determine_vmi13_area_ha(
            int(data_row[indices["county"]]),
            int(data_row[indices["lohkomuoto"]]),
            util.get_or_default(
                util.parse_int(data_row[indices["lohkotarkenne"]]),
                0)
        )
        result.area_weight_factors = vmi_util.determine_area_factors(
            data_row[indices["osuus4m"]],
            data_row[indices["osuus9m"]]
        )
        result.set_area(area_ha)
        lat = util.parse_type(data_row[indices["lat"]], float)
        lon = util.parse_type(data_row[indices["lon"]], float)
        height = vmi_util.transform_vmi13_height_above_sea_level(data_row[indices["height_above_sea_level"]])
        result.set_geo_location(lat, lon, height)
        result.drainage_year = vmi_util.determine_drainage_year(data_row[indices["ojitus_aika"]], result.year)
        result.fertilization_year = None  # value missing in VMI12 source
        result.soil_surface_preparation_year = vmi_util.determine_soil_surface_preparation_year(
            data_row[indices["maanmuokkaus"]],
            result.year
        )
        result.regeneration_area_cleaning_year = vmi_util.determine_clearing_of_reform_sector_year(
            data_row[indices["muu_toimenpide"]],
            data_row[indices["muu_toimenpide_aika"]],
            result.year)
        result.artificial_regeneration_year = vmi_util.determine_artificial_regeneration_year(
            data_row[indices["viljely"]],
            data_row[indices["viljely_aika"]],
            result.year)
        maintenance_details = vmi_util.determine_forest_maintenance_details(
            data_row[indices["hakkuu_tapa"]],
            data_row[indices["hakkuu_aika"]],
            result.year)
        result.young_stand_tending_year = maintenance_details[0]
        result.cutting_year = maintenance_details[1]
        result.method_of_last_cutting = maintenance_details[2]
        result.dominant_storey_age = vmi_util.determine_vmi13_dominant_storey_age(
            data_row[indices["vallitsevanjaksonika"]]
        )
        # Declared conversions
        result = self.conversion_reader.apply_conversions(result, data_row)
        return result

    def build(self) -> StandList:
        """ Populate a list of ForestStand with associated ReferenceTrees and TreeStrata in SoA form
        """
        result: dict[str, ForestStand] = {}
        # Per-stand attribute dicts
        strata_attrs: dict[str, dict[str, list]] = {}
        tree_attrs: dict[str, dict[str, list]] = {}

        # Build stands
        for i, row in enumerate(self.forest_stands):
            stand = self.convert_stand_entry(VMI13_STAND_INDICES, row, i + 1)
            result[stand.identifier] = stand

        # Strata → TreeStrata SoA
        if self.builder_flags.get('strata', False):
            for row in self.tree_strata:
                stand_identifier = vmi_util.generate_stand_identifier(row, VMI13_STRATUM_INDICES)
                attr_dict = strata_attrs.setdefault(stand_identifier, {})
                _append_stratum_row(attr_dict, VMI13_STRATUM_INDICES, row)

        # Trees → ReferenceTrees SoA
        if self.builder_flags.get('measured_trees', False):
            for row in self.reference_trees:
                stand_identifier = vmi_util.generate_stand_identifier(row, VMI13_TREE_INDICES)
                attr_dict = tree_attrs.setdefault(stand_identifier, {})
                _append_tree_row(attr_dict, VMI13_TREE_INDICES, row, is_vmi12=False)

        # Attach SoA containers to stands
        for stand_id, stand in result.items():
            stand.tree_strata = TreeStrata().vectorize(strata_attrs.get(stand_id, {}))
            stand.reference_trees = ReferenceTrees().vectorize(tree_attrs.get(stand_id, {}))

        return list(result.values())


class ForestCentreBuilder(ForestBuilder):
    ''' Base class for building a forest data model from Forest Centre (Suomen Metsakeskus) source '''

    @abstractmethod
    def build(self) -> StandList:
        ...

    @abstractmethod
    def convert_stand_entry(self, entry) -> ForestStand:
        ...


class XMLBuilder(ForestCentreBuilder):

    xpath_strata = './ts:TreeStandData/ts:TreeStandDataDate[@type="{}"]/tst:TreeStrata/tst:TreeStratum'
    xpath_stand = "st:Stands/st:Stand"

    def __init__(self, builder_flags: dict, declared_conversions: dict, data: str):
        self.root: ET.Element = ET.fromstring(data)
        self.builder_flags = builder_flags
        self.xpath_strata = self.xpath_strata.format(builder_flags['strata_origin'].value)
        self.declared_conversions = declared_conversions  # NOTE: not in use

    def set_stand_operations(self, stand: ForestStand, operations: dict[int, tuple[int, int]]) -> ForestStand:
        for oper in operations.values():
            (oper_type, oper_year) = oper
            if oper_type in (1,):
                stand.cutting_year = oper_year  # RST record 28
                stand.method_of_last_cutting = 4  # RST record 31
            elif oper_type in (2, 13, 20):
                stand.cutting_year = oper_year  # RST record 28
                stand.method_of_last_cutting = 3  # RST record 31
            elif oper_type in (3, 11, 12, 14, 91, 94):
                stand.cutting_year = oper_year  # RST record 28
                stand.method_of_last_cutting = 1  # RST record 31
            elif oper_type in (4, 15, 100):
                stand.cutting_year = oper_year  # RST record 28
                stand.method_of_last_cutting = 6 if stand.soil_peatland_category in (1, 2, 3) else 5
            elif oper_type in (6, 7, 102, 116, 123, 124):
                stand.cutting_year = oper_year  # RST record 28
                stand.method_of_last_cutting = 6  # RST record 31
            elif oper_type in (8, 101, 103, 104, 105, 106, 107, 108, 109,
                               110, 111, 112, 113, 114, 115, 117, 118,
                               119, 120, 121, 122, 125, 126, 127, 128):
                stand.cutting_year = oper_year  # RST record 28
                stand.method_of_last_cutting = 5  # RST record 31
            elif oper_type in (200, 201, 202, 203, 204, 205, 206, 207, 208,
                               209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
                               221, 222, 223, 224, 225, 226, 227, 228, 300, 301, 302, 303,
                               304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315,
                               316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327,
                               328, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611,
                               612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623,
                               624, 625, 626, 627, 628, 629, 630):
                stand.artificial_regeneration_year = oper_year  # RST record 25
            elif oper_type in (401, 410, 420, 450):
                stand.regeneration_area_cleaning_year = oper_year  # RST record 23
            elif oper_type in (501, 510, 511, 520, 521, 522, 523, 530, 531, 540, 550, 560, 960):
                stand.soil_surface_preparation_year = oper_year  # RST record 21
            elif oper_type in (660, 670, 680, 690, 701, 730, 740, 745, 750, 760, 860, 870, 880, 890):
                stand.young_stand_tending_year = oper_year  # RST record 26
            elif oper_type in (911, 912):
                stand.fertilization_year = oper_year  # RST record 20
            elif oper_type in (930, 940):
                stand.drainage_year = oper_year  # RST record 19

            else:
                print_logline(f'Unable to spesify operation type {oper_type} for stand \'{stand.identifier}\'')
                # raise UserWarning(f'Unable to spesify operation type {oper_type} for stand \'{stand.identifier}\'')
        return stand

    def convert_stand_entry(self, entry: ET.Element) -> ForestStand:
        stand_basic_data = smk_util.parse_stand_basic_data(entry)
        stand = ForestStand()
        stand.year = smk_util.parse_year(stand_basic_data.StandBasicDataDate)  # RST record 2
        stand.start_year = stand.year
        stand.set_area(util.parse_type(stand_basic_data.Area, float))  # RST record 3 and 4
        (latitude, longitude, crs) = smk_util.parse_coordinates(entry)
        stand.geo_location = (latitude, longitude, None, crs)  # RST record 5,6,8
        stand.identifier = stand_basic_data.id  # RST record 7
        stand.degree_days = None  # RST record 9
        stand.owner_category = OwnerCategory.PRIVATE  # RST record 10
        stand.land_use_category = fc2internal.convert_land_use_category(stand_basic_data.MainGroup)  # RST record 11
        stand.soil_peatland_category = fc2internal.convert_soil_peatland_category(
            stand_basic_data.SubGroup)  # RST record 12
        stand.site_type_category = fc2internal.convert_site_type_category(
            stand_basic_data.FertilityClass)  # RST record 13
        stand.tax_class_reduction = 0  # RST record 14
        stand.tax_class = 0  # RST record 15
        stand.drainage_category = fc2internal.convert_drainage_category(stand_basic_data.DrainageState)  # RST record 16
        # RST record 18 is '0' by default
        operations = smk_util.parse_stand_operations(entry, target_operations='past')
        stand = self.set_stand_operations(stand, operations)  # RST records 19, 20, 21, 23, 25, 26, 27, 28 and 31
        stand.development_class = smk_util.parse_development_class(0)  # RST record 24
        stand.forestry_centre_id = None  # RST record 29
        stand.forest_management_category = smk_util.parse_forest_management_category(
            stand_basic_data.CuttingRestriction) or 1  # 30
        stand.municipality_id = None  # RST record 32
        # RST record 33 and 34 unused
        stand.main_tree_species_dominant_storey = smk_util.determine_main_tree_species_dominant_storey(
            stand.site_type_category)
        return stand

    def build(self) -> StandList:
        stands = []
        estands = self.root.findall(self.xpath_stand, smk_util.NS)
        for estand in estands:
            stand = self.convert_stand_entry(estand)
            stratum_attr: dict[str, list] = {}

            estrata = estand.findall(self.xpath_strata, smk_util.NS)
            for estratum in estrata:
                _append_fc_stratum_row(stratum_attr, stand.identifier, estratum)

            stand.tree_strata = TreeStrata().vectorize(stratum_attr)
            stand.basal_area = float(np.nansum(stand.tree_strata.basal_area))
            stands.append(stand)
        return stands


class GeoPackageBuilder(ForestCentreBuilder):
    """ ForestBuilder for geopackage format spesification """
    stands: DataFrame
    strata: DataFrame
    type_value = None

    def __init__(self, builder_flags: dict, declared_conversions: dict, db_path: str):
        """ Reads Geopackage format into pandas dataframe representing stands and strata """
        self.type_value = builder_flags['strata_origin'].value
        (self.stands,
         self.strata) = gpkg_util.read_geopackage(db_path, self.type_value)
        self.declared_conversions = declared_conversions  # NOTE: not in use

    def convert_stand_entry(self, entry: Series) -> ForestStand:
        """ Converts a single pandas Series object into a ForestStand object
        :return: ForestStand object
        """
        stand = ForestStand()
        stand.year = smk_util.parse_year(entry.date)  # RST record 2
        stand.start_year = stand.year
        stand.set_area(entry.area - entry.areadecrease)  # RST record 3 and 4
        # RST records 5, 6 and 8
        (latitude, longitude) = entry.centroid.get('centroid')
        stand.geo_location = (latitude,
                              longitude,
                              None,
                              entry.centroid.get('crs'))
        stand.identifier = entry.standid  # RST record 7
        stand.degree_days = None  # RST record 9
        stand.owner_category = OwnerCategory.PRIVATE  # RST record 10
        stand.land_use_category = fc2internal.convert_land_use_category(
            util.parse_type(entry.maingroup, str))  # RST record 11
        stand.soil_peatland_category = fc2internal.convert_soil_peatland_category(
            util.parse_type(entry.subgroup, str))  # RST record 12
        stand.site_type_category = fc2internal.convert_site_type_category(
            util.parse_type(entry.fertilityclass, str))  # RST record 13
        # RST record 14
        # RST record 15
        stand.drainage_category = fc2internal.convert_to_internal(
            util.parse_type(entry.drainagestate, int, str),
            fc2internal.convert_drainage_category)  # RST record 16

        stand.development_class = smk_util.parse_development_class(
            util.parse_type(entry.developmentclass, str))  # RST record 24
        stand.forestry_centre_id = None  # RST record 29
        restrictioncode = entry.restrictioncode if entry.restrictiontype == 1 else 1
        stand.forest_management_category = smk_util.parse_forest_management_category(
            str(util.parse_type(restrictioncode, int, str)))  # 30
        stand.municipality_id = None  # RST record 32
        # RST record 33 and 34 unused
        stand.main_tree_species_dominant_storey = smk_util.determine_main_tree_species_dominant_storey(
            stand.site_type_category)
        return stand

    def build(self) -> StandList:
        """ Converts geopackage into list of ForestStand objects.
        :return: List of ForestStand objects
        """
        stands = []
        for _, rowi in self.stands.iterrows():
            stand = self.convert_stand_entry(rowi)
            stratum_attr: dict[str, list] = {}
            i_strata = self.strata[self.strata['standid'] == stand.identifier]
            for _, rowj in i_strata.iterrows():
                _append_gpkg_stratum_row(stratum_attr, stand.identifier, rowj)

            stand.tree_strata = TreeStrata().vectorize(stratum_attr)

            stand.basal_area = float(np.nansum(stand.tree_strata.basal_area))
            stands.append(stand)
        return stands
