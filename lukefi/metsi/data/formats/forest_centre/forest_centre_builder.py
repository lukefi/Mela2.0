from abc import abstractmethod
import xml.etree.ElementTree as ET
import numpy as np
from pandas import DataFrame, Series
from lukefi.metsi.app.console_logging import print_logline
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.conversion import fc2internal
from lukefi.metsi.data.enums.internal import CuttingMethod, OwnerCategory
from lukefi.metsi.data.formats import util
from lukefi.metsi.data.formats.forest_builder_base import ForestBuilder
from lukefi.metsi.data.formats.forest_centre import gpkg_util, smk_util
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import DTYPES_STRATA, TreeStrata
from lukefi.metsi.domain.forestry_types import StandList


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
                stand.method_of_last_cutting = CuttingMethod.OVER_STORY_REMOVAL  # RST record 31
            elif oper_type in (2, 13, 20):
                stand.cutting_year = oper_year  # RST record 28
                stand.method_of_last_cutting = CuttingMethod.FIRST_THINNING  # RST record 31
            elif oper_type in (3, 11, 12, 14, 91, 94):
                stand.cutting_year = oper_year  # RST record 28
                stand.method_of_last_cutting = CuttingMethod.THINNING  # RST record 31
            elif oper_type in (4, 15, 100):
                stand.cutting_year = oper_year  # RST record 28
                stand.method_of_last_cutting = CuttingMethod.SHELTERWOOD_CUTTING if stand.soil_peatland_category in (
                    1, 2, 3) else CuttingMethod.SEED_TREE_CUTTING
            elif oper_type in (6, 7, 102, 116, 123, 124):
                stand.cutting_year = oper_year  # RST record 28
                stand.method_of_last_cutting = CuttingMethod.SHELTERWOOD_CUTTING  # RST record 31
            elif oper_type in (8, 101, 103, 104, 105, 106, 107, 108, 109,
                               110, 111, 112, 113, 114, 115, 117, 118,
                               119, 120, 121, 122, 125, 126, 127, 128):
                stand.cutting_year = oper_year  # RST record 28
                stand.method_of_last_cutting = CuttingMethod.SEED_TREE_CUTTING  # RST record 31
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
        stand.development_class = None  # RST record 24
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
            try:
                stand = self.convert_stand_entry(estand)
                stratum_attr: dict[str, list] = {}

                estrata = estand.findall(self.xpath_strata, smk_util.NS)
                for estratum in estrata:
                    try:
                        _append_fc_stratum_row(stratum_attr, stand.identifier, estratum)
                    except Exception as e:
                        raise MetsiException(f"Parsing stratum {estratum} failed: {e}") from e

                stand.tree_strata = TreeStrata().vectorize(stratum_attr)
                stand.basal_area = float(np.nansum(stand.tree_strata.basal_area))
                stands.append(stand)
            except Exception as e:
                raise MetsiException(f"Parsing stand {estand} failed: {e}") from e

        return stands


def _append_fc_stratum_row(attr: dict[str, list], stand_identifier: str, estratum: ET.Element):
    """
    Append one Forest Centre (XML) stratum row into an SoA attribute dict.
    """

    sd = smk_util.parse_stratum_data(estratum)

    stratum_number = util.parse_type(sd.StratumNumber, int)
    raw_id = util.parse_type(sd.id, str)
    identifier = f"{stand_identifier}.{stratum_number or raw_id}-stratum"

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
        "stratum_number": stratum_number,
        "storey": fc2internal.convert_storey(sd.Storey),
        "sapling_stems_per_ha": 0.0,
        "number_of_generated_trees": None,
    }

    for key in DTYPES_STRATA:
        attr.setdefault(key, []).append(values.get(key, None))


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

        stand.development_class = None  # RST record 24
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
            try:
                stand = self.convert_stand_entry(rowi)
                stratum_attr: dict[str, list] = {}
                i_strata = self.strata[self.strata['standid'] == stand.identifier]
                for _, rowj in i_strata.iterrows():
                    try:
                        _append_gpkg_stratum_row(stratum_attr, stand.identifier, rowj)
                    except Exception as e:
                        raise MetsiException(f"Parsing stratum {rowj} failed: {e}") from e

                stand.tree_strata = TreeStrata().vectorize(stratum_attr)

                stand.basal_area = float(np.nansum(stand.tree_strata.basal_area))
                stands.append(stand)
            except Exception as e:
                raise MetsiException(f"Parsing stand {rowi} failed: {e}") from e

        return stands


def _append_gpkg_stratum_row(attr: dict[str, list], stand_identifier: str, rowj: Series):
    """Append one GeoPackage stratum row into an SoA attribute dict.
    """

    stratum_number = util.parse_type(rowj.stratumnumber, int)
    raw_id = util.parse_type(rowj.treestratumid, str)
    identifier = f"{stand_identifier}.{stratum_number or raw_id}-stratum"

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
        "stratum_number": stratum_number,
        "storey": util.parse_type(rowj.storey, int),
        "sapling_stems_per_ha": 0.0,
        "number_of_generated_trees": None,
    }

    for key in DTYPES_STRATA:
        attr.setdefault(key, []).append(values.get(key, None))
