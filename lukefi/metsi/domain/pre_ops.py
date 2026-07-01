from collections.abc import Callable
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.conversion import vmi2internal
from lukefi.metsi.data.enums.internal import (
    LandUseCategory,
    Storey,
    TreeCategory,
    TreeManagementCategory,
    TreeSpecies,
    TreeType)
from lukefi.metsi.data.enums.vmi import VmiIteration
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.domain.forestry_types import StandList
from lukefi.metsi.domain.utils.filter import filter_stands as filter_stands_
from lukefi.metsi.domain.utils.filter import filter_trees as filter_trees_
from lukefi.metsi.domain.utils.filter import filter_strata as filter_strata_
from lukefi.metsi.forestry.forestry_utils import find_matching_storey_stratum_for_tree
from lukefi.metsi.forestry.preprocessing.ages import ages
from lukefi.metsi.forestry.preprocessing.coordinate_conversion import convert_location_to_ykj, CRS
from lukefi.metsi.forestry.preprocessing.tree_generation import (
    adjust_ages, adjust_retention_trees, reference_trees_from_tree_stratum)


def filter_stands(stands: StandList,
                  *,
                  select: Optional[Callable[[ForestStand], bool]] = None,
                  remove: Optional[Callable[[ForestStand], bool]] = None) -> StandList:
    """Filter list of forest stands.

    Args:
        stands (StandList): list of stands to filter

    Returns:
        StandList: the filtered stands
    """
    if select is not None:
        stands = filter_stands_(stands, "select", select)
    if remove is not None:
        stands = filter_stands_(stands, "remove", remove)

    return stands


def filter_trees(stands: StandList,
                 *,
                 predicate: Callable[[ForestStand],
                                     npt.NDArray[np.bool_]] | None = None) -> StandList:
    """Filter reference trees for each stand in list based on given predicate.

    Args:
        stands (StandList): list of forest stands whose trees to filter
        predicate (Callable[[ForestStand], npt.NDArray[np.bool_]]): function that accepts a single stand and returns a
            numpy boolen array whose size matches the number of reference trees for the stand

    Returns:
        StandList: the list of stands with the trees filtered (also modified in-place)
    """
    assert predicate is not None
    stands = filter_trees_(stands, predicate)
    return stands


def filter_strata(stands: StandList, *, predicate: Callable[[ForestStand], npt.NDArray[np.bool_]]) -> StandList:
    """Filter tree strata for each stand in list based on given predicate. The predicate should be a

    Args:
        stands (StandList): list of forest stands whose strata to filter
        predicate (Callable[[ForestStand], npt.NDArray[np.bool_]]): function that accepts a single stand and returns a
            numpy boolen array whose size matches the number of strata for the stand

    Returns:
        StandList: the list of stands with the strata filtered (also modified in-place)
    """
    stands = filter_strata_(stands, predicate)
    return stands


def compute_location_metadata(stands: StandList, **operation_params) -> StandList:
    """
    This operation sets in-place the location based metadata properties for each given ForestStand, where missing.
    These properties are: height above sea level, temperature sum, sea effect, lake effect
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

    return stands


def generate_reference_trees(stands: StandList, /, **operation_params) -> StandList:
    """ Operation function that generates (N * stratum) reference trees for each stand """

    # oletusarvo true vai false?
    add_retention_trees = operation_params.get('add_retention_trees', True)

    stratum_association_diameter_threshold = operation_params.get('stratum_association_diameter_threshold', 3)

    for j, stand in enumerate(stands, 1):
        print(f"Generating trees for stand {stand.identifier}    {j}/{len(stands)}")

        tree_ordering = np.argsort(stand.reference_trees.identifier)

        stand.reference_trees = stand.reference_trees[tree_ordering]
        trees = stand.reference_trees

        strata = stand.tree_strata

        for i in range(len(trees)):
            matching_stratum = find_matching_storey_stratum_for_tree(
                trees.get_tree(i), strata, stratum_association_diameter_threshold)
            trees.stratum[i] = matching_stratum.stratum_number if matching_stratum is not None else -1

        retention_trees_mask = np.repeat(False, len(trees))
        if add_retention_trees:
            retention_trees_mask = (
                trees.stratum == -1) & (
                trees.management_category == TreeManagementCategory.RETENTION_TREE) & np.isin(
                trees.tree_type,
                (TreeType.UNSET,
                 TreeType.REMEASURED_TALLY_TREE,
                 TreeType.OLD_CHECKED_TALLY_TREE,
                 TreeType.NEW_TALLY_TREE_INCREMENT_HEIGHT_GREATER_THAN_1_3_M,
                 TreeType.NEW_TALLY_TREE_INCREMENT_HEIGHT_LESS_THAN_1_3_M,
                 TreeType.NEW_TALLY_TREE_OTHER_THAN_INCREMENT,
                 TreeType.OLD_TALLY_TREE_MEASURED_PREVIOUSLY_BY_MISTAKE)) & np.isin(
                trees.tree_category,
                (TreeCategory.UNSET,
                 TreeCategory.SMALL_TREE,
                 TreeCategory.WASTE_TREE,
                 TreeCategory.PULP_WOOD_TREE,
                 TreeCategory.SAW_LOG_TREE))

        stratum_ordering = np.argsort(stand.tree_strata.identifier)
        stand.tree_strata = stand.tree_strata[stratum_ordering]
        strata = stand.tree_strata

        new_trees = ReferenceTrees()

        for k, stratum in enumerate(strata.get_stratum(i) for i in range(len(strata))):
            try:
                stratum_trees = reference_trees_from_tree_stratum(stand, stratum, **operation_params)
                strata.number_of_generated_trees[k] = stratum.number_of_generated_trees
            except Exception as e:
                print(
                    f"\nError generating trees for stratum {
                        stratum.identifier} with diameter {
                        stratum.mean_diameter}, height {
                        stratum.mean_height}, basal_area {
                        stratum.basal_area}")
                print()
                raise e

            if stratum_trees is not None:
                stand_tree_count = len(new_trees)
                for i in range(len(stratum_trees)):
                    stratum_trees.identifier[i] = f"{stand.identifier}-{stand_tree_count + i + 1}-tree"

                new_trees = new_trees + stratum_trees

        # lisätään irralliset säästöpuut
        if add_retention_trees and np.any(retention_trees_mask):
            adjust_retention_trees(stand, new_trees, retention_trees_mask, operation_params['nfi_iteration'])
        if operation_params.get('age_model', False):
            for i in range(len(new_trees)):
                if new_trees.breast_height_diameter[i] > 0:
                    breast_height_age, biological_age = ages(
                        stand, new_trees + stand.reference_trees[retention_trees_mask], new_trees.get_tree(i), 10)
                    new_trees.breast_height_age[i] = round(breast_height_age, 1)
                    new_trees.biological_age[i] = round(biological_age, 1)
            adjust_ages(stand, new_trees)

        retention_trees = trees[retention_trees_mask]

        new_strata = TreeStrata(retention_trees.size)
        new_strata.stratum_number = np.arange(1, len(retention_trees) + 1, dtype=np.int32) + len(stand.tree_strata)
        retention_trees.stratum = new_strata.stratum_number
        new_strata.identifier = np.asarray([
            stand.identifier +
            "-" +
            str(stratum_number) +
            "-stratum" for stratum_number in new_strata.stratum_number])
        new_strata.species = retention_trees.species
        new_strata.origin = retention_trees.origin
        new_strata.mean_diameter = retention_trees.breast_height_diameter
        new_strata.mean_height = retention_trees.height
        new_strata.breast_height_age = retention_trees.breast_height_age
        new_strata.biological_age = retention_trees.biological_age
        new_strata.storey = np.repeat(Storey.SPARE, len(retention_trees))
        new_strata.stems_per_ha = retention_trees.stems_per_ha
        new_strata.basal_area = retention_trees.stems_per_ha * np.pi * \
            ((retention_trees.breast_height_diameter / 200) ** 2)
        new_strata.number_of_generated_trees = np.repeat(1, len(retention_trees))

        stand.tree_strata = stand.tree_strata + new_strata

        stand.reference_trees = new_trees + retention_trees

        if operation_params.get("delete_strata", False):
            stand.tree_strata = TreeStrata()

    return stands


def scale_basal_area_at_county_level(stands: StandList, *, nfi_iteration: VmiIteration | None = None) -> StandList:
    """Scale basal area at the county/forestry centre level to match basal areas by species in NFI data. County is used
       for NFI iterations 12 and up, forestry centre for earlier.
       NOTE: It is supposed that all stands belong to same county (or forestry centre) and represent the whole county
       (or fc).

    Args:
        stands (StandList): the list of stands to update

    Returns:
        StandList: updated stands
    """

    assert nfi_iteration is not None

    county = stands[0].region
    if county == 19 and stands[1].municipality_id in (47, 148, 890):
        county = 30

    # basal area sums by species an land use classes (index 0 = forest land, 1 = scrub land)
    ba_sums = np.asarray([[0.0] * max(TreeSpecies), [0.0] * max(TreeSpecies)], dtype=np.float64)
    ba_sum_ret = 0  # retention trees

    for stand in stands:
        assert stand.land_use_category is not None

        if stand.land_use_category not in (LandUseCategory.FOREST, LandUseCategory.SCRUB_LAND):
            continue

        trees = stand.reference_trees
        bhd_positive = trees.breast_height_diameter > 0
        is_retained = trees.management_category == TreeManagementCategory.RETENTION_TREE
        is_not_retained = ~is_retained

        for species in TreeSpecies:
            if species in (TreeSpecies.UNSET, TreeSpecies.TREELESS):
                continue
            mask1 = bhd_positive & is_not_retained & (trees.species == species)
            mask2 = bhd_positive & is_retained & (trees.species == species)

            ba_sums[stand.land_use_category - 1][species - 1] += stand.area * np.pi * \
                np.sum(trees.stems_per_ha[mask1] * ((trees.breast_height_diameter[mask1] / 200) ** 2))
            ba_sum_ret += stand.area * np.pi * \
                np.sum(trees.stems_per_ha[mask2] * ((trees.breast_height_diameter[mask2] / 200) ** 2))

    # scale coefficients
    forest_land_ba = pd.read_csv(f'lukefi/metsi/data/nfi_data/{nfi_iteration.upper()}/PPA_metsamaa.csv',
                                 sep=' ',
                                 index_col="maakunta")
    scrub_land_ba = pd.read_csv(f'lukefi/metsi/data/nfi_data/{nfi_iteration.upper()}/PPA_kitumaa.csv',
                                sep=' ',
                                index_col="maakunta")
    retention_trees_ba = pd.read_csv(f'lukefi/metsi/data/nfi_data/{nfi_iteration.upper()}/PPA_saastopuut.csv',
                                     sep=' ',
                                     index_col="maakunta")

    ba_targets = [np.full(max(TreeSpecies), 0.0, dtype=np.float64),
                  np.full(max(TreeSpecies), 0.0, dtype=np.float64)]

    geo_index = stands[0].forestry_centre_id if nfi_iteration in (
        VmiIteration.VMI9, VmiIteration.VMI10, VmiIteration.VMI11) else county

    assert geo_index is not None

    for species_col in forest_land_ba:
        ba_targets[0][vmi2internal.convert_species(str(species_col)) - 1] = forest_land_ba[species_col][geo_index]
    for species_col in scrub_land_ba:
        ba_targets[1][vmi2internal.convert_species(str(species_col)) - 1] = scrub_land_ba[species_col][geo_index]

    ba_target_ret: np.float64 = retention_trees_ba.V2[geo_index]

    if len(ba_targets) == 0:
        scale_coeffs = [[1] * max(TreeSpecies), [1] * max(TreeSpecies)]
    else:
        scale_coeffs = [[], []]
        for i in range(2):
            for target, generated in zip(ba_targets[i], ba_sums[i]):
                coeff = target / generated if generated > 0 else -1
                scale_coeffs[i].append(coeff)
    scale_coeff_ret = ba_target_ret / ba_sum_ret if ba_sum_ret > 0 else -1

    for stand in stands:
        assert stand.land_use_category is not None

        if stand.land_use_category not in (LandUseCategory.FOREST, LandUseCategory.SCRUB_LAND):
            continue

        trees = stand.reference_trees
        is_retained = trees.management_category == TreeManagementCategory.RETENTION_TREE
        is_not_retained = ~is_retained
        scale_coeffs_for_trees = np.asarray(scale_coeffs[stand.land_use_category - 1])[trees.species - 1]
        mask = is_not_retained & (scale_coeffs_for_trees >= 0)

        trees.stems_per_ha[mask] *= scale_coeffs_for_trees[mask]

        if scale_coeff_ret >= 0:
            trees.stems_per_ha[is_retained] *= scale_coeff_ret

    return stands


def update_strata_to_match_trees(stands: StandList, **operation_params) -> StandList:
    _ = operation_params

    for stand in stands:
        i_tree = -1
        stand_ba = 0

        trees = stand.reference_trees
        strata = stand.tree_strata

        for i_stratum in range(len(strata)):
            ntrees = 0
            stems = 0
            stratum_ba = 0
            species_ba = [0] * max(TreeSpecies)
            dsum = 0
            hsum = 0
            hasum = 0
            aged13sum = 0
            aged13asum = 0
            agebiolsum = 0
            agebiolasum = 0

            # count the number of reference_trees left (having positive stems_per_ha) for each stratum
            for _ in range(strata.number_of_generated_trees[i_stratum]):
                i_tree = i_tree + 1
                if trees.stems_per_ha[i_tree] > 0:
                    tree_ba = trees.stems_per_ha[i_tree] * np.pi * ((trees.breast_height_diameter[i_tree] / 200)**2)
                    dsum = dsum + tree_ba * trees.breast_height_diameter[i_tree]
                    hsum = hsum + tree_ba * trees.height[i_tree]
                    hasum = hasum + trees.stems_per_ha[i_tree] * trees.height[i_tree]
                    aged13sum = aged13sum + tree_ba * trees.breast_height_age[i_tree]
                    aged13asum = aged13asum + trees.stems_per_ha[i_tree] * trees.breast_height_age[i_tree]
                    agebiolsum = agebiolsum + tree_ba * trees.biological_age[i_tree]
                    agebiolasum = agebiolasum + trees.stems_per_ha[i_tree] * trees.biological_age[i_tree]
                    stratum_ba = stratum_ba + tree_ba
                    species_ba[trees.species[i_tree] - 1] = species_ba[trees.species[i_tree] - 1] + tree_ba

                    ntrees = ntrees + 1
                    stems = stems + trees.stems_per_ha[i_tree]

            # update stratum
            strata.basal_area[i_stratum] = stratum_ba
            if stems > 0:
                strata.mean_diameter[i_stratum] = dsum / stratum_ba if stratum_ba > 0 else 0
                strata.mean_height[i_stratum] = hsum / stratum_ba if stratum_ba > 0 else hasum / stems
                strata.breast_height_age[i_stratum] = aged13sum / stratum_ba if stratum_ba > 0 else aged13asum / stems
                strata.biological_age[i_stratum] = agebiolsum / stratum_ba if stratum_ba > 0 else agebiolasum / stems
            else:
                strata.mean_diameter[i_stratum] = np.nan
                strata.mean_height[i_stratum] = np.nan
                strata.breast_height_age[i_stratum] = np.nan
                strata.biological_age[i_stratum] = np.nan
            strata.stems_per_ha[i_stratum] = stems
            strata.number_of_generated_trees[i_stratum] = ntrees
            strata.sapling_stems_per_ha[i_stratum] = None

            # basal area of the whole stand
            stand_ba += stratum_ba

        trees.delete(np.where(trees.stems_per_ha < 0.00005)[0])

        stand.basal_area = stand_ba
    return stands


def scale_area_weight(stands: StandList, /, **operation_params):
    """ Scales area weight of a stand.

        Especially necessary for VMI tree generation cases.
        Should be used as precesing operation before the generation of reference trees.
    """
    _ = operation_params
    for stand in stands:
        stand.area_weight = stand.area_weight * stand.area_weight_factors[1]
    return stands


def area_ha_to_1000ha(stands: StandList, **operation_params):
    # Converts area of a stand from ha to 1000 ha
    _ = operation_params
    for stand in stands:
        stand.area_weight = stand.area_weight / 1000
        stand.area = stand.area / 1000
    return stands


def scale_trees_by_area_weight_factors(stands: StandList, **operation_params):
    """Scale the number of stems of the (measured) trees according to the stand's
       proportion of the sample plot. Trees with diameter in [4.5,9.5) are scaled
       by proportion of the sample plot having 4 m radius. Trees having diameter >= 9.5 cm
       are scaled by proportion of the sample plot with 9 m radius.

    Args:
        stands (StandList): list of ForestStands

    Returns:
        StandList: the modified stands
    """
    _ = operation_params
    for stand in stands:
        trees = stand.reference_trees
        if len(trees) > 0:
            smaller_diameter = ((4.5 <= trees.breast_height_diameter) &
                                (trees.breast_height_diameter < 9.5) &
                                (0 < stand.area_weight_factors[0] < 1))
            larger_diameter = (trees.breast_height_diameter >= 9.5) & (0 < stand.area_weight_factors[1] < 1)
            trees.stems_per_ha[smaller_diameter] = trees.stems_per_ha[smaller_diameter] / stand.area_weight_factors[0]
            trees.stems_per_ha[larger_diameter] = trees.stems_per_ha[larger_diameter] / stand.area_weight_factors[1]

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


__all__ = ['filter_stands',
           'filter_trees',
           'filter_strata',
           'compute_location_metadata',
           'generate_reference_trees',
           'scale_basal_area_at_county_level',
           'update_strata_to_match_trees',
           'scale_area_weight',
           'area_ha_to_1000ha',
           'scale_trees_by_area_weight_factors',
           'convert_coordinates']
