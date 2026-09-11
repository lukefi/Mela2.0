from typing import Optional
import math
from dataclasses import dataclass
from datetime import datetime as dt

from lukefi.metsi.data.enums.internal import SiteType, Storey, StratumRank, TreeManagementCategory, TreeSpecies
from lukefi.metsi.data.enums.vmi import VmiIteration
from lukefi.metsi.data.formats.util import get_or_default, parse_float, parse_int, parse_type
from lukefi.metsi.core.exceptions import MetsiException


def generate_source_data(indices: dict[str, slice], row: str) -> dict[str, str]:
    """
    Create a dictionary of source data values from raw data row and index description.
    If the data row does not contain an index (i.e. the row is cut short)
    an empty string is returned natively when indexing by the slice.
    """
    return {key: row[index] for key, index in indices.items()}


def determine_area_factors(small_tree_sourcevalue: str, big_tree_sourcevalue: str) -> tuple[float, float]:
    """Compute forest stand specific scaling factors for area and reference tree stem count scaling."""
    small = get_or_default(parse_float(small_tree_sourcevalue), 0.0) / 10
    big = get_or_default(parse_float(big_tree_sourcevalue), 0.0) / 10
    return small, big


def determine_artificial_regeneration_year(regeneration: str, regeneration_year: str, year: int) -> Optional[int]:
    if regeneration in ('1', '2', '3', '4'):
        if regeneration_year == '0':
            return year
        if regeneration_year == '1':
            return year - 1
        if regeneration_year == '2':
            return year - 3
        if regeneration_year == '3':
            return year - 8
        if regeneration_year in ('a', 'A'):
            return year - 20
        if regeneration_year in ('b', 'B'):
            return year - 35
    return None


def determine_main_tree_species_dominant_storey(species_source: str,
                                                site_type_category: Optional[SiteType]) -> Optional[TreeSpecies]:
    if species_source in [' ', '.']:
        return None
    parsed_int = parse_int(species_source)
    if parsed_int is None:
        return None
    if parsed_int == 0:
        if site_type_category is None:
            return None
        if site_type_category <= SiteType.DAMP_SITE:
            return TreeSpecies.SPRUCE
        return TreeSpecies.PINE
    return TreeSpecies(parsed_int)


def determine_clearing_of_reform_sector_year(other_method: str, year_adjustment_class: str, year: int) -> Optional[int]:
    """Determine the year of reform sector clearing when "other method" matches with the correct class in VMI terms"""
    if other_method == '4':
        if year_adjustment_class == "0":
            return year
        if year_adjustment_class == "1":
            return year - 1
        if year_adjustment_class == "2":
            return year - 3
        if year_adjustment_class == "3":
            return year - 8
    return None


def determine_drainage_year(sourcevalue: str, year: int) -> Optional[int]:
    try:
        value = int(sourcevalue)
        return year - value
    except (TypeError, ValueError):
        return None


def determine_soil_surface_preparation_year(sourcevalue: str, year: int) -> Optional[int]:
    """Determine the year of soil surface preparation from given VMI source classes and the year of data set."""
    if sourcevalue == '0':
        return year
    if sourcevalue == '1':
        return year - 1
    if sourcevalue == '2':
        return year - 3
    if sourcevalue == '3':
        return year - 8
    if sourcevalue in {'A', 'a'}:
        return year - 20
    return None


def determine_tax_class_reduction(sourcevalue: str) -> int:
    """
    Map and return number valued source string as integer for values 0 to 4. Otherwise 0.
    """
    if sourcevalue == '0':
        return 0
    if sourcevalue == '1':
        return 1
    if sourcevalue == '2':
        return 2
    if sourcevalue == '3':
        return 3
    if sourcevalue == '4':
        return 4
    return 0


def determine_tax_class(sourcevalue: str) -> int:
    """
    Map and return number valued source string as int for values [0,4] => [1,5]. Otherwise 0.
    """

    if sourcevalue == '0':
        return 1
    if sourcevalue == '1':
        return 2
    if sourcevalue == '2':
        return 3
    if sourcevalue == '3':
        return 4
    if sourcevalue == '4':
        return 5
    return 0


def determine_forest_management_category(land_use_category: int,
                                         is_ahvenanmaa: bool,
                                         owner_group: int,
                                         production_limitation: str,
                                         production_limitation_detail: str,
                                         other_values: str,
                                         protection_forest_code: str,
                                         aland_area_code: str,
                                         test_area_handling_class: str

                                         ) -> float:
    # Determine forest management category  for given conditions.
    # Determine first the NFI management category (vmi_pt)
    # MELA management category is then determined by NFI management category and land_use_category
    vmi_pt = 3
    decimals = 10.0

    if (production_limitation in ('101',
                                  '102',
                                  '103',
                                  '104',
                                  '105',
                                  '108',
                                  '301',
                                  '401',
                                  '402',
                                  '403',
                                  '404',
                                  '408',
                                  '409')):
        vmi_pt = 1
        decimals = int(production_limitation) / 1000

    if (production_limitation in ('107',
                                  '109',
                                  '201',
                                  '205',
                                  '206',
                                  '207',
                                  '302',
                                  '303',
                                  '304',
                                  '305',
                                  '306',
                                  '307',
                                  '308',
                                  '309',
                                  '310',
                                  '405',
                                  '406',
                                  '407',
                                  '501',
                                  '502',
                                  '503',
                                  '504') and production_limitation_detail in ('1', '2') and vmi_pt == 3):
        vmi_pt = 1
        decimals = int(production_limitation) / 1000

    if (production_limitation in ('107',
                                  '109',
                                  '201',
                                  '205',
                                  '206',
                                  '207',
                                  '302',
                                  '303',
                                  '304',
                                  '305',
                                  '306',
                                  '307',
                                  '308',
                                  '309',
                                  '310',
                                  '405',
                                  '406',
                                  '407',
                                  '501',
                                  '502',
                                  '503',
                                  '504') and production_limitation_detail in ('3', '4') and vmi_pt == 3):
        vmi_pt = 2
        decimals = int(production_limitation) / 1000

    if (production_limitation in ('202', '203') and
            production_limitation_detail in ('1', '2', '3', '4') and vmi_pt == 3):
        vmi_pt = 2
        decimals = int(production_limitation) / 1000

    if other_values in ('1', '2', '3', '4', '5', '6') and vmi_pt == 3:
        vmi_pt = 2
        decimals = 0.6 + int(other_values) / 100

    if land_use_category in (2, 3) and vmi_pt == 3:
        vmi_pt = 2
        decimals = 0.9

    if protection_forest_code == '1' and owner_group == 4 and vmi_pt == 3:
        vmi_pt = 2
        decimals = 0.7

    # Ahvenanmaa;
    if (is_ahvenanmaa and (aland_area_code != '1' or other_values == '2') and vmi_pt > 1):
        vmi_pt = 1
        decimals = 0.7

    # Metsähallituksen rajoitukset;
    try:
        test_area_handling_class_numeric = float(test_area_handling_class)
        if test_area_handling_class_numeric == 1:
            mh_pt = 3
        elif test_area_handling_class_numeric == 2:
            mh_pt = 2
        elif test_area_handling_class_numeric in (3.1, 3.2):
            mh_pt = 1
        else:
            mh_pt = 5
    except ValueError:
        mh_pt = 5

    if mh_pt < vmi_pt:
        decimals = 0.8

    pt = min(vmi_pt, mh_pt)

    # MELA forest management category
    fmc = 1.0
    if pt == 3 and land_use_category == 1:
        fmc = 1
    if pt == 3 and land_use_category == 2:
        fmc = 3
    if pt == 3 and land_use_category == 3:
        fmc = 6

    if pt == 2 and land_use_category == 1:
        fmc = 2
    if pt == 2 and land_use_category == 2:
        fmc = 3
    if pt == 2 and land_use_category == 3:
        fmc = 6

    if pt == 1:
        fmc = 7

    # decimals
    if decimals < 1 < fmc:
        fmc = fmc + decimals

    return fmc


def determine_tree_age_values(chest_height_age_source: str, age_increase_source: str,
                              total_age_source: str) -> tuple[int | None, int | None]:
    chest_height_age = parse_int(chest_height_age_source)
    age_increase = parse_int(age_increase_source)
    total_age = parse_int(total_age_source)

    if total_age:
        computed_age = total_age
    elif age_increase and chest_height_age:
        computed_age = chest_height_age + age_increase
    elif chest_height_age:
        computed_age = chest_height_age + 9
    else:
        computed_age = None

    return None if chest_height_age == 0 else chest_height_age, computed_age


def determine_tree_management_category(latvuskerros: str) -> TreeManagementCategory:
    return TreeManagementCategory.RETENTION_TREE \
        if latvuskerros.lower() in ('b', 'c', 'd', 'e', 'f', 'g') \
        else TreeManagementCategory.NO_RESTRICTION


def determine_tree_height(height_sourcevalue: str, conversion_factor: float = 10.0) -> Optional[float]:
    """
    return tree height in meters as transformed from VMI dm values or computed with the Näslund height model
    if VMI value is not available.

    :param height_sourcevalue: integer string assumed to represent decimeters
    :param diameter:
    :param species:
    :return:
    """
    h = get_or_default(parse_float(vmi_codevalue(height_sourcevalue)), 0.0)
    return h / conversion_factor if h > 0 else None


def determine_stems_per_ha(
    diameter_cm: float,
    vmi_version: VmiIteration = VmiIteration.VMI13,
    forestry_centre_id: int | None = None,
    ahvkeilaus: str | None = None,
) -> float:
    """
    stems_per_ha logic for VMI9..VMI13.
    """

    p = get_stems_params(vmi_version, forestry_centre_id, ahvkeilaus)

    d = float(diameter_cm)
    if d <= 0.0:
        return 1.0

    if d < p.d1:
        n = 40000.0 * p.q / (math.pi * d * d)
        # keep legacy-ish rounding for small trees in 12/13
        return float(round(n, 5))

    if d < p.d2:
        n = 10000.0 / (math.pi * p.r1 * p.r1)
        return float(round(n, 5))

    n = 10000.0 / (math.pi * p.r2 * p.r2)
    return float(round(n, 5))


def determine_stratum_tree_height(source_height: str) -> Optional[float]:
    maybe_height = parse_float(source_height)
    if maybe_height is not None and maybe_height > 0:
        return round(maybe_height / 10, 2)
    return None


def determine_stratum_age_values(biological_age_source: str,
                                 breast_height_age_source: str,
                                 height: Optional[float]) -> tuple[float, float]:
    """
    Determinates biological age and breast height age for vmi source data.

    param: biological_age_source: Stratum biological age or age increase value as vmi source value.
    param: breast_height_age_source: Stratum breast height age as vmi source value
    param: height (optional): Stratum height value

    return: Biological and breast height age as a tuple of whole number floats
    """
    computational_age = get_or_default(parse_float(biological_age_source), 0.0)
    breast_height_age = get_or_default(parse_float(breast_height_age_source), 0.0)
    if height is None:
        height = 0.0

    if computational_age == 0 and breast_height_age > 0:
        computational_age = breast_height_age + 9
    elif computational_age == 0 and breast_height_age == 0 and height > 0:
        if height > 1.3:
            breast_height_age = 1.4 * height
            computational_age = breast_height_age + 8
            breast_height_age = round(breast_height_age, 0)
            computational_age = round(computational_age, 0)
        else:
            # sapling biological age
            computational_age = 1.4 * height
            computational_age = round(computational_age, 0)
    elif computational_age > 0 and breast_height_age == 0 and height > 1.3:
        breast_height_age = 1.4 * height
        computational_age = breast_height_age + computational_age
        breast_height_age = round(breast_height_age, 0)
        computational_age = round(computational_age, 0)
    elif computational_age > 0:
        computational_age = computational_age + breast_height_age
    else:
        computational_age = 0.0

    return (computational_age, breast_height_age)


def determine_storey_for_stratum(source: StratumRank) -> Storey:
    """Determinates storey for stratum based on vmi source value 'ositteen asema'."""
    if source in [StratumRank.UNPRODUCTIVE_SEEDLINGS, StratumRank.DOMINANT_TREE_STOREY]:
        return Storey.DOMINANT

    if source in [StratumRank.OVER_STOREY, StratumRank.NURSE_CROP]:
        return Storey.OVER

    if source in [
            StratumRank.UNDER_STOREY_DEVELOPMENT_CAPABLE,
            StratumRank.UNDER_STOREY_NOT_DEVELOPMENT_CAPABLE,
            StratumRank.NON_ESTABLISHED_SEEDLINGS,
            StratumRank.SEEDLING_STRATUM]:
        return Storey.UNDER

    if source in [StratumRank.DAMAGED_TREE_STRATUM]:
        return Storey.INDETERMINATE

    if source == StratumRank.RETENTION_TREE_STOREY:
        return Storey.RETENTION

    return Storey.INDETERMINATE


def determine_storey_for_tree(source: str) -> Optional[Storey]:
    """Determinates storey for vmi tree based on vmi source value 'latvuskerros'."""
    parsed = parse_int(source)
    if parsed in [2, 3, 4]:
        return Storey.DOMINANT
    if parsed in [5]:
        return Storey.UNDER
    if parsed in [6, 7]:
        return Storey.OVER
    return None


def determine_municipality(municipality_code: str, kitukunta: str) -> Optional[int]:
    """
    Return by order of precedence: valid municipality code, valid kitukunta code, or None.
    """
    retval = parse_int(vmi_codevalue(municipality_code))
    if retval is None:
        retval = parse_int(vmi_codevalue(kitukunta))
    return retval


def parse_date(date_string: str) -> dt:
    """Generate a datetime entry out of VMI12 date source format ddmmyy"""
    parsed = dt.strptime(date_string, '%d%m%y')
    return apply_growth_inc_logic(parsed)


def apply_growth_inc_logic(date_obj: dt) -> dt:
    """If month >= 7, increment year by 1. (yearly growth is over)"""
    if date_obj.month >= 7:
        return date_obj.replace(year=date_obj.year + 1)
    return date_obj


def parse_forestry_centre(forestry_centre: str) -> int:
    try:
        return int(forestry_centre)
    except (ValueError, TypeError):
        return 10


def parse_vmi_area_ha(raw: str) -> float:
    s = (raw or "").strip()
    if not s:
        return 0.0
    if "." in s:
        return get_or_default(parse_type(s, float), 0.0)
    # implied 5 decimals
    try:
        return int(s) / 100000.0
    except ValueError:
        return get_or_default(parse_type(s, float), 0.0)


def transform_vmi_degree_days(sourcevalue: str) -> float | None:
    """Return given number value string as float or None on error"""
    try:
        return float(sourcevalue)
    except ValueError:
        return None


def transform_tree_diameter(source: str) -> float:
    return get_or_default(parse_float(source), 0.0) / 10.0


def vmi_codevalue(source: str) -> Optional[str]:
    value = source.strip()
    if value in ('', '.'):
        return None
    return value


def generate_stand_identifier(source_data: dict[str, str]) -> str:
    return source_data["lohkomuoto"] + "-" + \
        source_data["section_y"] + "-" + \
        source_data["section_x"] + "-" + \
        source_data["test_area_number"] + "-" + \
        source_data["stand_number"]


def generate_tree_identifier(source_data: dict[str, str]) -> str:
    return source_data["lohkomuoto"] + "-" + \
        source_data["section_y"] + "-" + \
        source_data["section_x"] + "-" + \
        source_data["test_area_number"] + "-" + \
        source_data["stand_number"] + "-" + \
        source_data["tree_number"] + "-" + \
        "tree"


def generate_stratum_identifier(source_data: dict[str, str]) -> str:
    return source_data["lohkomuoto"] + "-" + \
        source_data["section_y"] + "-" + \
        source_data["section_x"] + "-" + \
        source_data["test_area_number"] + "-" + \
        source_data["stand_number"] + "-" + \
        source_data["stratum_number"] + "-" + \
        "stratum"


def parse_share_tenths(raw: str) -> float:
    """Share is coded 0..10 meaning 0.0..1.0"""
    s = get_or_default(parse_float((raw or "").strip()), 0.0)
    return max(0.0, min(1.0, s / 10.0))


def parse_int0(raw: str) -> int:
    return get_or_default(parse_int((raw or "").strip()), 0)


def parse_float0(raw: str) -> float:
    return get_or_default(parse_float((raw or "").strip()), 0.0)


@dataclass(frozen=True)
class StemsParams:
    q: float
    r1: float
    r2: float
    d1: float
    d2: float


def _is_north_finland(forestry_centre_id: Optional[int]) -> bool:
    if forestry_centre_id is None:
        return False
    return 11 <= forestry_centre_id <= 13


def _is_ahvenanmaa(forestry_centre_id: Optional[int], ahvkeilaus: Optional[str]) -> bool:
    if forestry_centre_id == 0 and ahvkeilaus == 'A':
        return True
    return False


def get_stems_params(
        vmi_version: VmiIteration,
        forestry_centre_id: Optional[int],
        ahvkeilaus: Optional[str]) -> StemsParams:
    """
    Parameters from the provided R-document.
    forestry_centre_id:
      0          => Ahvenanmaa
      0..10      => Etelä-Suomi
      11..13     => Pohjois-Suomi
    """

    if vmi_version == VmiIteration.VMI13:
        return StemsParams(q=1.5, r1=4.0, r2=9.0, d1=4.5, d2=9.5)

    if vmi_version == VmiIteration.VMI12:
        return StemsParams(q=1.5, r1=5.64, r2=9.0, d1=4.5, d2=9.5)

    if vmi_version == VmiIteration.VMI11 and _is_ahvenanmaa(forestry_centre_id, ahvkeilaus):
        return StemsParams(q=1.0, r1=9.0, r2=9.0, d1=18.0, d2=9999.0)

    if vmi_version in (VmiIteration.VMI9, VmiIteration.VMI10, VmiIteration.VMI11):
        if _is_north_finland(forestry_centre_id):
            return StemsParams(q=1.5, r1=12.45, r2=12.45, d1=30.49615, d2=9999.0)
        # South (includes Ahvenanmaa for 9/10 and non-special 11)
        return StemsParams(q=2.0, r1=12.52, r2=12.52, d1=35.41191, d2=9999.0)

    raise MetsiException(f"Unsupported VMI version for stems_per_ha: {vmi_version}")
