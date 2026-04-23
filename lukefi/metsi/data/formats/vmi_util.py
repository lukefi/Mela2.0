from typing import Optional
import math
from dataclasses import dataclass
from datetime import datetime as dt
from shapely.geometry import Point
from geopandas import GeoSeries

from lukefi.metsi.data.enums.internal import Origin, SiteType, Storey, TreeManagementCategory, TreeSpecies
from lukefi.metsi.data.enums.vmi import VmiIteration
from lukefi.metsi.data.formats.util import get_or_default, parse_float, parse_int, parse_type
from lukefi.metsi.data.conversion import vmi2internal

from lukefi.metsi.data.formats.vmi_const import VMI12_COUNTY_AREAS, VMI10_COUNTY_AREAS, VMI11_COUNTY_AREAS
from lukefi.metsi.app.utils import MetsiException


def _solve_vmi13_county_areas(county: int, lohkomuoto: int, lohkotarkenne: int) -> float:
    if county == 1 and lohkomuoto == 2 and lohkotarkenne == 0:
        return 345.73918
    if county == 2 and lohkomuoto == 2 and lohkotarkenne == 0:
        return 338.0386443
    if county == 4 and lohkomuoto == 2 and lohkotarkenne == 0:
        return 342.975010960105
    if county == 5 and lohkomuoto == 2 and lohkotarkenne == 0:
        return 342.747528
    if county == 6 and lohkotarkenne == 0:
        if lohkomuoto == 1:
            return 413.08125
        if lohkomuoto == 2:
            return 347.828958275767
    if county == 7 and lohkomuoto == 2 and lohkotarkenne == 0:
        return 342.438585979628
    if county == 8 and lohkomuoto == 2 and lohkotarkenne == 0:
        return 349.917881811205
    if county == 9 and lohkomuoto == 2 and lohkotarkenne == 0:
        return 350.8972332
    if county == 10 and lohkomuoto == 2 and lohkotarkenne == 0:
        return 340.4779333
    if county == 11:
        if lohkomuoto == 1:
            return 436.521343
        if lohkomuoto == 2:
            return 330.3735632
    if county == 12 and lohkotarkenne == 0:
        if lohkomuoto == 1:
            return 433.4836506
        if lohkomuoto == 2:
            return 351.5358362
    if county == 13 and lohkomuoto == 1 and lohkotarkenne == 0:
        return 435.9383152
    if county == 14 and lohkotarkenne == 0:
        if lohkomuoto == 1:
            return 429.5909091
        if lohkomuoto == 2:
            return 351.5358362
    if county == 15 and lohkomuoto == 1 and lohkotarkenne == 0:
        return 434.9541716
    if county == 16 and lohkomuoto == 1 and lohkotarkenne == 0:
        return 435.0433276
    if county == 17 and lohkotarkenne == 0:
        if lohkomuoto == 1:
            return 435.0433276
        if lohkomuoto == 3:
            return 457.7258227
        if lohkomuoto == 4:
            return 747.6246246
    if county == 18 and lohkomuoto == 3 and lohkotarkenne == 0:
        return 455.8440533
    if county == 19:
        if lohkomuoto == 4:
            if lohkotarkenne == 0:
                return 786.978534
        if lohkomuoto == 5:
            if lohkotarkenne == 0:
                return 1357.608776
            if lohkotarkenne == 1:
                return 1176.023409
            if lohkotarkenne == 2:
                return 1355.455959
            if lohkotarkenne == 3:
                return 1999.800742
            if lohkotarkenne == 4:
                return 10756.11645
    if county == 21 and lohkomuoto == 0 and lohkotarkenne == 0:
        return 164.2650475

    raise MetsiException(f"Unable to solve vmi13 country area weight for values: \
                        county {county}, lohkomuoto {lohkomuoto} and lohkotarkenne {lohkotarkenne}")


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


def determine_natural_renewal(natural_renewal: str) -> bool:
    return natural_renewal.strip() in {'8', '9'}


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


def determine_drainage_feasibility(ojitus_tarve: str) -> bool:
    return ojitus_tarve in ('1', '2', '3')


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


def determine_vmi12_dominant_storey_age(ds_bh_age: str, ds_age_increase: str) -> float:
    """ Dominant storey age is composed of dominant storey breast height age and age increase for vmi12. """
    a = get_or_default(parse_float(ds_bh_age), 0.0)
    b = get_or_default(parse_float(ds_age_increase), 0.0)
    return a + b


def determine_vmi13_dominant_storey_age(ds_age) -> float:
    """ Dominant storey mean age for vmi13 """
    return get_or_default(parse_float(ds_age), 0.0)


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


def determine_stratum_origin_vmi9(source_origin: str) -> Origin:

    if source_origin in ("1", "2"):
        return Origin.NATURAL
    if source_origin in ("3", "5", "7", "8"):
        return Origin.PLANTED
    if source_origin in ("4", "6"):
        return Origin.SEEDED
    return Origin.NATURAL


def determine_stratum_origin_vmi10(source_origin: str) -> Origin:

    if source_origin in ("0", "1", "2"):
        return Origin.NATURAL
    if source_origin == "3":
        return Origin.PLANTED
    if source_origin == "4":
        return Origin.SEEDED
    return Origin.NATURAL


def determine_stratum_age_values(biological_age_source: str,
                                 breast_height_age_source: str,
                                 height: Optional[float]) -> tuple[float, float]:
    """ Determinates biological age and breast height age for vmi source data.

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


def determine_storey_for_stratum(source: str) -> Optional[Storey]:
    """Determinates storey for stratum based on vmi source value 'ositteen asema'."""
    parsed = parse_int(source)
    if parsed in [0, 1]:
        return Storey.DOMINANT
    if parsed in [2, 3, 4]:
        return Storey.OVER
    if parsed in [5, 6, 7, 9]:
        return Storey.UNDER
    if parsed in [8]:
        return Storey.INDETERMINATE
    return None


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


def determine_tree_type(source: str) -> Optional[str]:
    if source in (' ', '.', ''):
        return None
    return source


def determine_municipality(municipality_code: str, kitukunta: str) -> Optional[int]:
    """
    Return by order of precedence: valid municipality code, valid kitukunta code, or None.
    """
    retval = parse_int(vmi_codevalue(municipality_code))
    if retval is None:
        retval = parse_int(vmi_codevalue(kitukunta))
    return retval


def convert_vmi12_geolocation(lat_source: str, lon_source: str) -> tuple[float, float]:
    """
    Convert VMI12 coordinates in EPSG:2393 to EPSG:3067. Source values are in meter precision, return values are
    likewise rounded to meter precision.
    :param lat_source: EPSG:2393 latitude
    :param lon_source: EPSG:2393 longitude
    :return: lat, lon tuple in EPSG:3067
    """
    point = GeoSeries([Point(float(lon_source), float(lat_source))], crs='EPSG:2393')
    point = point.to_crs(3067)
    return round(point.centroid.y[0]), round(point.centroid.x[0])


def convert_vmi12_approximate_geolocation(lat_source: str, lon_source: str) -> tuple[float, float]:
    """
    Convert VMI12 coordinates in EPSG:2393 to YKJ/KKJ3 with band 3 prefix removed.

    :param lat_source: source string of the latitude value
    :param lon_source: source string of the llongitude value

    :return (lat,lon): latitude,longitude pair
    """
    lat = float(lat_source)
    lon = float(lon_source) - 3000000
    return lat, lon


def parse_vmi12_date(date_string: str) -> dt:
    """Generate a datetime entry out of VMI12 date source format ddmmyy"""
    parsed = dt.strptime(date_string, '%d%m%y')
    return _apply_growth_inc_logic(parsed)


def parse_vmi13_date(date_string: str) -> dt:
    """Generate a datetime entry out of VMI13 date source format yyyymmdd"""
    parsed = dt.strptime(date_string, '%Y%m%d')
    return _apply_growth_inc_logic(parsed)


def _apply_growth_inc_logic(date_obj: dt) -> dt:
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


def get_vmi11_area_ha(forestry_centre: int, osite: int) -> float:
    """
    VMI11 area lookup.
    Returns area_ha for given metkes (forestry_centre) and osite (lohkomuoto / 300 / 400).
    """
    try:
        return VMI11_COUNTY_AREAS[forestry_centre][osite]
    except KeyError as exc:
        raise KeyError(
            f"No VMI11 area_ha for metkes={forestry_centre}, osite={osite}"
        ) from exc


def determine_vmi12_area_ha(lohkomuoto: int, county: int) -> float:
    area_ha = 0.0
    if county < 1 or county >= len(VMI12_COUNTY_AREAS):
        raise IndexError
    if county < 17:
        area_ha = VMI12_COUNTY_AREAS[(county - 1)]
    elif county == 17 and lohkomuoto == 3:
        area_ha = VMI12_COUNTY_AREAS[(county - 1)]
    elif county == 17 and lohkomuoto == 4:
        area_ha = VMI12_COUNTY_AREAS[county]
    elif county == 18:
        area_ha = VMI12_COUNTY_AREAS[18]
    elif county == 19 and lohkomuoto == 4:
        area_ha = VMI12_COUNTY_AREAS[19]
    elif county == 19 and lohkomuoto == 5:
        area_ha = VMI12_COUNTY_AREAS[20]
    elif county == 21:
        area_ha = VMI12_COUNTY_AREAS[21]
    return round(area_ha, 4)


def determine_vmi13_area_ha(county: int, lohkomuoto: int, lohkotarkenne: int) -> float:
    if county < 0 and lohkomuoto < 0 or lohkotarkenne < 0:
        raise IndexError
    return _solve_vmi13_county_areas(county, lohkomuoto, lohkotarkenne)


def transform_vmi12_height_above_sea_level(sourcevalue: str) -> float | None:
    """
    Transform given VMI12 number value string from desimeters to meters.
    Returning float, or None on error.
    """
    try:
        return float(sourcevalue) / 10.0
    except ValueError:
        return None


def transform_vmi13_height_above_sea_level(sourcevalue: str) -> float | None:
    """Return given number value string as float, or None on error"""
    try:
        return float(sourcevalue)
    except ValueError:
        return None


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


def append_tree_row_vmi9(attr: dict[str, list], indices, row: str, forestry_centre_id: int | None):
    """
    Append one VMI9 tree row into SoA dict compatible with DTYPES_TREE.
    """
    identifier = generate_tree_identifier(row, indices)
    tree_number = get_or_default(parse_type(row[indices["tree_number"]], int), 0)

    species = vmi2internal.convert_species(row[indices["species"]])

    raw_tc = row[indices["tree_category"]].strip()
    tc_enum = vmi2internal.convert_tree_category(raw_tc)

    tree_category = tc_enum.value if tc_enum else None
    breast_height_diameter = transform_tree_diameter(row[indices["diameter"]])

    # dm -> m
    height = determine_tree_height(row[indices["height"]], conversion_factor=10.0)
    measured_height = None

    lowest_living_branch_height = (
        get_or_default(parse_type(row[indices["living_branches_height"]], float), 0.0) / 10.0
    )

    breast_height_age, biological_age = determine_tree_age_values(
        row[indices["d13_age"]],
        row[indices["age_increase"]],
        row[indices["total_age"]],
    )

    stems_per_ha = determine_stems_per_ha(breast_height_diameter, vmi_version=VmiIteration.VMI9,
                                          forestry_centre_id=forestry_centre_id)

    management_category = determine_tree_management_category(row[indices["latvuskerros"]])
    storey = determine_storey_for_tree(row[indices["latvuskerros"]])

    tuhon_raw = row[indices["tuhon_ilmiasu"]]
    damage_type = None if tuhon_raw in (" ", ".", "") else tuhon_raw.strip()

    values = {
        "identifier": identifier,
        "tree_number": tree_number,
        "species": species,
        "tree_category": tree_category,
        "breast_height_diameter": breast_height_diameter,
        "height": height,
        "measured_height": measured_height,
        "breast_height_age": breast_height_age,
        "biological_age": biological_age,
        "stems_per_ha": stems_per_ha,
        "origin": 0,
        "management_category": management_category,
        "storey": storey,
        "saw_log_volume_reduction_factor": None,
        "pruning_year": 0,
        "age_when_10cm_diameter_at_breast_height": 0,
        "stand_origin_relative_position": (0.0, 0.0, 0.0),
        "lowest_living_branch_height": lowest_living_branch_height,
        "sapling": False,
        "tree_type": None,
        "damage_type": damage_type,
        "basal_area": None,
    }

    for k, v in values.items():
        attr.setdefault(k, []).append(v)


def determine_storey_for_segment(asema_raw: str) -> Storey:
    """
    VMI10 jakson asema:
      1 Vallitseva jakso
      2 Ylispuusto
      3 Jättöylispuusto
      4 Verhopuusto
      5 Kehityskelpoinen alikasvos
      6 Kehityskelvoton alikasvos
      7 Vaihtuva taimiaines
    """
    v = (asema_raw or "").strip()
    if not v or v == ".":
        return Storey.INDETERMINATE
    if v == "1":
        return Storey.DOMINANT
    if v == "2":
        return Storey.OVER
    if v == "3":
        return Storey.REMOVAL
    if v == "4":
        return Storey.SPARE
    if v == "5":
        return Storey.UNDER
    if v == "6":
        return Storey.REMOTE
    if v == "7":
        return Storey.INDETERMINATE
    return Storey.INDETERMINATE


def parse_share_tenths(raw: str) -> float:
    """Share is coded 0..10 meaning 0.0..1.0"""
    s = get_or_default(parse_float((raw or "").strip()), 0.0)
    return max(0.0, min(1.0, s / 10.0))


def parse_int0(raw: str) -> int:
    return get_or_default(parse_int((raw or "").strip()), 0)


def parse_float0(raw: str) -> float:
    return get_or_default(parse_float((raw or "").strip()), 0.0)

def append_vmi9_strata_from_stand_row(
    attr: dict[str, list],
    indices: dict[str, slice],
    stand_row: str,
    stand_identifier: str,
    stand_basal_area: float,
):
    """
    Build up to 6 strata (2 segments x (main + up to 2 side species)) into TreeStrata
    """

    fallback_age = parse_float0(stand_row[indices["metsikon_ika"]])

    jakso2_ppa = parse_float0(stand_row[indices["jakso2_ppa"]])
    jakso1_ppa = max(0.0, stand_basal_area - jakso2_ppa)

    running_numb = 0

    def emit_stratum_vmi9(
        species_code_raw: str,
        share_raw: str,
        seg_stems1000_raw: str,
        seg_d_cm_raw: str,
        seg_h_dm_raw: str,
        seg_d13_age_raw: str,
        seg_age_inc_raw: str,
        seg_syntytapa_raw: str,
        seg_asema_raw: str,
        seg_ppa_total: float,
    ):
        nonlocal running_numb
        species_code = (species_code_raw or "").strip()
        if not species_code or species_code in (".", "0"):
            return

        share = parse_share_tenths(share_raw)
        if share <= 0.0:
            return

        species = vmi2internal.convert_species(species_code)

        basal_area = seg_ppa_total * share
        stems_per_ha = parse_float0(seg_stems1000_raw) * 1000.0 * share
        mean_diameter = parse_float0(seg_d_cm_raw)
        mean_height = parse_float0(seg_h_dm_raw) / 10.0  # dm -> m
        age = _vmi10_segment_age_years(seg_d13_age_raw, seg_age_inc_raw, fallback_age=fallback_age)

        syntytapa = determine_stratum_origin_vmi9(seg_syntytapa_raw)
        storey = determine_storey_for_segment(seg_asema_raw)

        running_numb += 1
        identifier = f"{stand_identifier}-{running_numb}-stratum"

        values = {
            "identifier": identifier,
            "species": int(species),
            "mean_diameter": mean_diameter,
            "mean_height": mean_height,
            "breast_height_age": parse_int0(seg_d13_age_raw),
            "biological_age": age,
            "stems_per_ha": stems_per_ha,
            "basal_area": basal_area,
            "origin": syntytapa,
            "management_category": 0,
            "saw_log_volume_reduction_factor": None,
            "cutting_year": 0,
            "age_when_10cm_diameter_at_breast_height": 0,
            "stratum_number": running_numb,
            "stand_origin_relative_position": (0.0, 0.0, 0.0),
            "lowest_living_branch_height": 0.0,
            "storey": int(storey),
            "sapling_stems_per_ha": 0.0,
            "sapling_stratum": False,
            "number_of_generated_trees": 0,
        }

        for k, v in values.items():
            attr.setdefault(k, []).append(v)

    for seg_no in (1, 2):
        ppa_total = jakso1_ppa if seg_no == 1 else jakso2_ppa

        asema = stand_row[indices[f"jakso{seg_no}_asema"]]
        synty = stand_row[indices[f"jakso{seg_no}_syntytapa"]]

        stems1000 = stand_row[indices[f"jakso{seg_no}_kokonaisrunkoluku1000"]]
        d_cm = stand_row[indices[f"jakso{seg_no}_keskilapimitta_cm"]]
        h_dm = stand_row[indices[f"jakso{seg_no}_keskipituus_dm"]]
        d13ika = stand_row[indices[f"jakso{seg_no}_d13ika"]]
        ikalis = stand_row[indices[f"jakso{seg_no}_ikalisays"]]

        main_share_raw = stand_row[indices[f"jakso{seg_no}_paapuulaji_osuus"]]
        siv1_share_raw = stand_row[indices[f"jakso{seg_no}_sivulaji1_osuus"]]

        main_t = parse_int0(main_share_raw)
        siv1_t = parse_int0(siv1_share_raw)

        siv2_t = max(0, 10 - main_t - siv1_t)

        emit_stratum_vmi9(
            stand_row[indices[f"jakso{seg_no}_paapuulaji"]],
            str(main_t),
            stems1000, d_cm, h_dm, d13ika, ikalis, synty, asema, ppa_total
        )

        emit_stratum_vmi9(
            stand_row[indices[f"jakso{seg_no}_sivulaji1"]],
            str(siv1_t),
            stems1000, d_cm, h_dm, d13ika, ikalis, synty, asema, ppa_total
        )

        emit_stratum_vmi9(
            stand_row[indices[f"jakso{seg_no}_sivulaji2"]],
            str(siv2_t),
            stems1000, d_cm, h_dm, d13ika, ikalis, synty, asema, ppa_total
        )


def determine_vmi11_area_ha(
    forestry_centre: int,
    lohkomuoto: int,
    eduala_raw: str,
    inventointitunnus: str | None = None,
    lohy_raw: str | None = None,
    ahvkeilaus: str | None = None,
) -> float:
    """
    Determine stand area_ha for VMI11.

    Default:
      - area_ha = eduala with 5 decimals

    Lookup:
      - Normal lohkomuoto osite in {1,2,3,4} -> lookup from VMI11_COUNTY_AREAS
      - Ahvenanmaa special osite in {300,400} -> lookup from VMI11_COUNTY_AREAS (metkes=0 row)

    Ahvenanmaa rules:
      - inventointitunnus = P and ahvkeilaus = A => osite 300 => area 100.39
      - inventointitunnus = P and ahvkeilaus = B => osite 400 => area 148.78
      - inventointitunnus = K and lohy(1:1) = 3 => osite 300 => area 100.39
      - inventointitunnus = K and lohy(1:1) = 4 => osite 400 => area 148.78
    """
    default_area_ha = parse_vmi_area_ha(eduala_raw)

    inv = (inventointitunnus or "").strip().upper()
    ak = (ahvkeilaus or "").strip().upper()
    lohy_first = ((lohy_raw or "").strip()[:1] or "")

    if inv == "P" and ak in ("A", "B"):
        osite = 300 if ak == "A" else 400
        return round(get_vmi11_area_ha(0, osite), 4)

    if inv == "K" and lohy_first in ("3", "4"):
        osite = 300 if lohy_first == "3" else 400
        return round(get_vmi11_area_ha(0, osite), 4)

    osite = lohkomuoto
    use_lookup = (1 <= osite <= 4) or (forestry_centre == 0)

    if not use_lookup:
        return round(default_area_ha, 4)

    try:
        return round(get_vmi11_area_ha(forestry_centre, osite), 4)
    except KeyError as exc:
        raise MetsiException(
            f"No area_ha lookup value for VMI11: metkes={forestry_centre}, osite={osite}. "
            f"inventointitunnus={inv!r}, lohy={lohy_raw!r}, ahvkeilaus={ak!r}, lohkomuoto={lohkomuoto}."
        ) from exc


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


def determine_forest_management_category_vmi9(
    stand_row: str,
    indices: dict[str, slice],
) -> float:
    """
    VMI9 käsittelyluokka (forest_management_category) calculation.
    """

    owner = parse_int0(stand_row[indices["owner_group"]])
    ptraj = parse_int0(stand_row[indices["ptraj"]])
    pttark = parse_int0(stand_row[indices["pttark"]])
    land_category = parse_int0(stand_row[indices["land_category"]])

    ml123_ala = parse_float0(stand_row[indices["ml123ala"]])

    abt1 = parse_int0(stand_row[indices["abi1kasehd"]])
    abt1_ala = parse_float0(stand_row[indices["abi1ala"]])

    abt2 = parse_int0(stand_row[indices["abi2kasehd"]])
    abt2_ala = parse_float0(stand_row[indices["abi2ala"]])

    abt3 = parse_int0(stand_row[indices["abi3kasehd"]])
    abt3_ala = parse_float0(stand_row[indices["abi3ala"]])

    # mhptrajtar is only meaningful in North Finland; in South your indices are slice(0,0) or value is blank.
    mhtark = parse_int0(stand_row[indices["mhptrajtar"]]) if "mhptrajtar" in indices else 0

    # ------------------------------------------------------------
    # Start
    k = 1.0

    # --- avainbiotoopit ---
    avainbt = max(abt1, abt2, abt3)

    avainbt_pinta_ala = 0.0
    avainbt_ala = 0.0

    if avainbt == 2:
        avainbt_pinta_ala = abt1_ala + abt2_ala + abt3_ala

    if ml123_ala > 0.0:
        avainbt_ala = avainbt_pinta_ala / ml123_ala

    # --- metsähallituksen rajoituksen tarkennus ---
    if owner == 4:
        ptraj = 0

    if owner == 4 and mhtark in (2, 3, 4, 5, 6, 9):
        ptraj = mhtark

    # --- käsittelyluokka ---
    if ptraj in (101, 102):
        k = 7.1

    if ptraj in (401, 402, 403, 404, 408, 409, 410):
        k = 7.2

    if ptraj == 105 and owner not in (4, 5):
        k = 7.3

    if ptraj == 105 and owner in (4, 5):
        k = 7.5

    if ptraj in (104, 106):
        k = 7.5

    if ptraj == 103:
        k = 7.4

    if ptraj in (201, 205, 301):
        k = 7.5

    if ptraj == 405:
        k = 2.1

    if ptraj in (202, 203, 307):
        k = 2.4

    if ptraj == 304:
        k = 2.2

    if ptraj == 303:
        k = 2.3

    if ptraj == 501 and pttark in (1, 2, 3, 4, 5) and owner in (4, 5):
        k = 2.4

    if ptraj in (309, 306, 302):
        k = 2.4

    if ptraj == 107 and pttark == 1:
        k = 7.5

    if ptraj == 107 and pttark != 1:
        k = 2.4

    if ptraj == 606 and pttark == 1 and avainbt == 2:
        k = 7.5

    if ptraj == 606 and pttark != 1 and avainbt == 2:
        k = 2.4

    if ptraj == 608 and pttark == 1 and avainbt == 2 and avainbt_ala >= 0.5:
        k = 7.5

    if ptraj == 608 and pttark != 1 and avainbt == 2 and avainbt_ala >= 0.5:
        k = 2.4

    if ptraj == 601 and owner == 4:
        k = 2.4

    if ptraj == 305:
        k = 2.4

    if owner == 4 and ptraj == 0:
        k = 1.0

    if owner == 4 and ptraj in (2, 3, 4, 5):
        k = 2.4

    if owner == 4 and ptraj in (6, 9):
        k = 7.5

    # --- maaluokka adjustments ---
    if land_category == 2 and k == 1.0:
        k = 3.5

    if land_category == 2 and 2.0 <= k < 3.0:
        k = k + 1.0

    if land_category == 3 and k == 1.0:
        k = 6.5

    if land_category == 3 and 2.0 <= k < 3.0:
        k = k + 4.0

    if land_category == 3 and k > 7.0:
        k = 7.6

    return float(k)
