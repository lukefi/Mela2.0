from typing import Any, Callable
from lukefi.metsi.domain.forestry_types import StandList
from lukefi.metsi.domain.utils.filter import applyfilter
from lukefi.metsi.domain.utils.opt_utils import opt_species
from lukefi.metsi.forestry.preprocessing import tree_generation, pre_util
from lukefi.metsi.forestry.preprocessing.coordinate_conversion import convert_location_to_ykj, CRS
from lukefi.metsi.forestry.preprocessing.age_supplementing import supplement_age_for_reference_trees
from lukefi.metsi.forestry.preprocessing.naslund import naslund_height
from lukefi.metsi.app.utils import MetsiException


def preproc_filter(stands: StandList, **operation_params) -> StandList:
    command: str
    predicate: Callable[..., bool]
    for command, predicate in operation_params.items():
        stands = applyfilter(stands, command, predicate)
    return stands


def compute_location_metadata(stands: StandList, **operation_params) -> StandList:
    """
    This operation sets in-place the location based metadata properties for each given ForestStand, where missing.
    These properties are: height above sea level, temperature sum, sea effect, lake effect, monthly temperature and
    monthly rainfall
    """

    # Lazy import of optional MetsiGrow functions.
    from lukefi.metsi.forestry.naturalprocess.MetsiGrow.metsi_grow.lasum import ilmanor  # pylint: disable=import-outside-toplevel
    from lukefi.metsi.forestry.naturalprocess.MetsiGrow.metsi_grow.coord import etrs_tm35_to_ykj as conv  # pylint: disable=import-outside-toplevel
    from lukefi.metsi.forestry.naturalprocess.MetsiGrow.metsi_grow.kor import xkor  # pylint: disable=import-outside-toplevel

    _ = operation_params

    for stand in stands:

        if stand.geo_location is None:
            raise MetsiException(f"Stand {stand.identifier} has no geolocation data")
        if stand.geo_location[0] is None or stand.geo_location[1] is None:
            raise MetsiException(f"Stand {stand.identifier} has incomplete geolocation data: {stand.geo_location}")

        if stand.geo_location[3] == 'EPSG:3067':
            lat, lon = conv(stand.geo_location[0] / 1000, stand.geo_location[1] / 1000)
        elif stand.geo_location[3] == 'EPSG:2393':
            lat, lon = (stand.geo_location[0] / 1000, stand.geo_location[1] / 1000)
        else:
            raise MetsiException(f"Unsupported CRS {stand.geo_location[3]} for stand {stand.identifier}")

        if stand.geo_location[2] is None:
            xkor_value = xkor(lat, lon)
            stand.geo_location = (
                stand.geo_location[0],
                stand.geo_location[1],
                xkor_value,
                stand.geo_location[3])
        else:
            xkor_value = stand.geo_location[2]

        wi = ilmanor(lon, lat, xkor_value)

        if stand.degree_days is None:
            stand.degree_days = wi.dd
        if stand.sea_effect is None:
            stand.sea_effect = wi.sea
        if stand.lake_effect is None:
            stand.lake_effect = wi.lake
        if stand.monthly_temperatures is None:
            stand.monthly_temperatures = wi.temp
        if stand.monthly_rainfall is None:
            stand.monthly_rainfall = wi.rain

    return stands


def generate_reference_trees(stands: StandList, **operation_params) -> StandList:
    """ Operation function that generates reference trees for each stand """
    debug = operation_params.get("debug", False)

    for i, stand in enumerate(stands):
        print(f"\rGenerating trees for stand {stand.identifier}    {i}/{len(stands)}", end="")

        try:
            new_vec = tree_generation.generate_reference_trees(stand, **operation_params)
        except Exception as e:  # noqa: BLE001
            print(f"\nError generating trees for stand {stand.identifier}")
            if debug:
                # pylint: disable=import-outside-toplevel
                import traceback  # type: ignore[import-outside-toplevel]
                traceback.print_exc()
                continue
            raise e

        stand.reference_trees = new_vec

    print()
    return stands


def supplement_missing_tree_heights(stands: StandList, **operation_params) -> StandList:
    """Fill in missing (None or nonpositive) tree heights from Näslund height curve using SoA data."""
    _ = operation_params
    for stand in stands:
        trees = stand.reference_trees
        if trees.size == 0:
            continue

        for i in range(trees.size):
            h = trees.height[i]

            if not h > 0:
                d = trees.breast_height_diameter[i]
                if not d > 0:
                    continue

                species = opt_species(trees.species[i])
                new_h = naslund_height(float(d), species)
                if new_h is None:
                    continue

                trees.update({"height": float(new_h)}, i)

    return stands


def supplement_missing_tree_ages(stands: StandList, **operation_params):
    """
    Supplement tree ages directly on SoA ReferenceTrees/TreeStrata.
    """
    _ = operation_params
    for stand in stands:
        trees_vec = stand.reference_trees
        strata_vec = stand.tree_strata

        if trees_vec.size == 0 or strata_vec.size == 0:
            continue

        supplement_age_for_reference_trees(trees_vec, strata_vec)

    return stands


def supplement_missing_stratum_diameters(stands: StandList, **operation_params) -> StandList:
    _ = operation_params  # unused

    for stand in stands:
        if stand.tree_strata.size == 0:
            continue
        pre_util.supplement_mean_diameter(
            stand.tree_strata,
            _land_use_category=stand.land_use_category,
        )
    return stands


def generate_sapling_trees_from_sapling_strata(stands: StandList, **operation_params) -> StandList:
    _ = operation_params  # unused

    for stand in stands:
        strata = stand.tree_strata
        trees = stand.reference_trees
        base = trees.size

        rows = pre_util.create_sapling_rows_from_strata(strata, stand.identifier, base)
        if rows:
            trees.create(rows)
    return stands


def scale_area_weight(stands: StandList, **operation_params):
    """ Scales area weight of a stand.

        Especially necessary for VMI tree generation cases.
        Should be used as precesing operation before the generation of reference trees.
    """
    _ = operation_params
    for stand in stands:
        stand.area_weight = stand.area_weight * stand.area_weight_factors[1]
    return stands


def convert_coordinates(stands: StandList, **operation_params: dict[str, Any]) -> StandList:
    """ Preprocessing operation for converting the current coordinate system to target system

    :target_system (optional): Spesified target system. Default is EPSG:2393
    """
    defaults = CRS.EPSG_2393.value
    target_system = operation_params.get('target_system', defaults[0])
    if target_system in defaults:
        for s in stands:
            if s.geo_location is not None:
                latitude, longitude, height, crs = s.geo_location
                if latitude is not None and longitude is not None:
                    s.geo_location = convert_location_to_ykj(latitude, longitude, height, crs)
    else:
        raise MetsiException("Check definition of operation params.\n"
                             f"{defaults[0]}\' conversion supported.")
    return stands


__all__ = ['preproc_filter',
           'compute_location_metadata',
           'generate_reference_trees',
           'supplement_missing_tree_heights',
           'supplement_missing_tree_ages',
           'supplement_missing_stratum_diameters',
           'generate_sapling_trees_from_sapling_strata',
           'scale_area_weight',
           'convert_coordinates',
           'vectorize']
