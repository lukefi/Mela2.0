from typing import Any, Callable

import numpy as np
from lukefi.metsi.data.ba_nfi import BA_NFI, BA_NFI_RET
from lukefi.metsi.data.enums.internal import TreeSpecies
from lukefi.metsi.domain.forestry_types import StandList
from lukefi.metsi.domain.utils.filter import applyfilter
from lukefi.metsi.forestry.preprocessing import tree_generation
from lukefi.metsi.forestry.preprocessing.coordinate_conversion import convert_location_to_ykj, CRS
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


def generate_reference_trees(stands: StandList, **operation_params) -> StandList:
    """ Operation function that generates (N * stratum) reference trees for each stand """
    debug = operation_params.get('debug', False)
    debug_output_rows = []
    debug_strata_rows = []
    debug_tree_rows = []

    # oletusarvo true vai false?
    add_retention_trees = operation_params.get('add_retention_trees', True)

    stratum_association_diameter_threshold = operation_params.get('stratum_association_diameter_threshold', 3)
    for i, stand in enumerate(stands):
        print(f"\rGenerating trees for stand {stand.identifier}    {i}/{len(stands)}", end="")
        retention_trees = []
        stand_trees = sorted(stand.reference_trees_pre_vec, key=lambda tree: tree.identifier)
        for tree in stand_trees:

            stratum = find_matching_storey_stratum_for_tree(
                tree, stand.tree_strata, stratum_association_diameter_threshold)
            if stratum is None:
                if add_retention_trees and tree.management_category == 2 and tree.is_measured_type() and tree.is_living():
                    # ositteisiin kuulumattomat säästöpuut
                    retention_trees.append(tree)
                continue

            if stratum.__dict__.get('_trees') is not None:
                stratum._trees.append(tree)
            else:
                stratum._trees = [tree]
            if debug:
                debug_tree_rows.append([
                    stratum.identifier,
                    tree.species.value,
                    tree.tree_category,
                    tree.tree_type,
                    tree.latvuskerros,
                    tree.breast_height_diameter or 'NA',
                    tree.measured_height or 'NA',
                    tree.stems_per_ha or 'NA'
                ])
        stand.tree_strata.sort(key=lambda stratum: stratum.identifier)
        new_trees = []
        for stratum in stand.tree_strata:
            stratum_trees = []
            try:
                stratum_trees = tree_generation.reference_trees_from_tree_stratum(stratum, **operation_params)
            except Exception as e:
                print(
                    f"\nError generating trees for stratum {
                        stratum.identifier} with diameter {
                        stratum.mean_diameter}, height {
                        stratum.mean_height}, basal_area {
                        stratum.basal_area}")
                print()
                if debug:
                    traceback.print_exc()
                    continue
                else:
                    raise e
            stand_tree_count = len(new_trees)
            for i, tree in enumerate(stratum_trees):
                tree.identifier = "{}-{}-tree".format(stand.identifier, stand_tree_count + i + 1)
                new_trees.append(tree)

            validation_set = create_stratum_tree_comparison_set(stratum, stratum_trees)

            if debug:
                debug_strata_rows.append([
                    stratum.identifier,
                    stratum.mean_diameter,
                    stratum.mean_height,
                    stand.basal_area,
                    stratum.basal_area,
                    stratum.species.value,
                    stand.degree_days
                ])
                debug_output_rows.append(debug_output_row_from_comparison_set(stratum, validation_set))

        # lisätään irralliset säästöpuut
        if len(retention_trees) > 0:
            retention_trees, new_trees = adjust_retention_trees(stand, new_trees, retention_trees)
        if operation_params.get('age_model', False):
            for t in new_trees:
                if t.breast_height_diameter > 0:
                    breast_height_age, biological_age = ages(stand, t, 10, new_trees + retention_trees)
                    t.breast_height_age = round(breast_height_age, 1)
                    t.biological_age = round(biological_age, 1)
            adjust_ages(stand, new_trees)

        for t in retention_trees:
            new_trees.append(t)
            new_stratum = generate_stratum_from_retention_tree(t, stand.identifier, len(stand.tree_strata))
            stand.tree_strata.append(new_stratum)

        stand.reference_trees = [rt for rt in new_trees]

    print()
    if debug:
        import csv
        with open('debug_generated_tree_results.csv', 'w', newline='\n') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(debug_output_header_row())
            writer.writerows(debug_output_rows)
        if len(debug_strata_rows) > 1:
            with open('r_strata.dat', 'w', newline='\n') as stratum_file:
                writer = csv.writer(stratum_file, delimiter=' ')
                writer.writerow(["stratum", "DGM", "HGM", "G", "Gos", "spe", "DDY"])
                writer.writerows(debug_strata_rows)
        if len(debug_tree_rows) > 1:
            with open('r_trees.dat', 'w', newline='\n') as tree_file:
                writer = csv.writer(tree_file, delimiter=' ')
                writer.writerow(["stratum", "spe", "cat", "type", "latker", "lpm", "height", "lkm"])
                writer.writerows(debug_tree_rows)

    return stands


def scale_basal_area_at_county_level(stands: StandList, **operation_params) -> StandList:
    # scale basal area at the county level to match basal areas by species in NFI data
    # NOTE: It is supposed that all stands belong to same county and represent the whole county

    # Basal areas by county, land use categories and species

    _ = operation_params

    county = stands[0].region
    if county == 19 and stands[1].municipality_id in (47, 148, 890):
        county = 30

    # basal area sums by species an land use classes (index 0 = forest land, 1 = scrub land)
    ba_sums = np.asarray([[0] * max(TreeSpecies), [0] * max(TreeSpecies)])
    ba_sum_ret = 0  # retention trees

    for stand in stands:
        assert stand.land_use_category is not None

        trees = stand.reference_trees
        bhd_positive = trees.breast_height_diameter > 0
        is_retained = trees.management_category == 2
        is_not_retained = ~is_retained

        for species in TreeSpecies:
            mask1 = bhd_positive & is_not_retained & (trees.species == species)
            mask2 = bhd_positive & is_retained & (trees.species == species)

            ba_sums[stand.land_use_category - 1][species - 1] += stand.area * np.pi * \
                np.sum(trees.stems_per_ha[mask1] * ((trees.breast_height_diameter[mask1] / 200) ** 2))
            ba_sum_ret += stand.area * np.pi * \
                np.sum(trees.stems_per_ha[mask2] * ((trees.breast_height_diameter[mask2] / 200) ** 2))

    # scale coefficients
    ba_targets: list[list[list[float]]] = [[x.ppa for x in BA_NFI if x.maakunta == county and x.maalk == 1],
                                           [x.ppa for x in BA_NFI if x.maakunta == county and x.maalk == 2]]
    ba_target_ret = [x.ppa for x in BA_NFI_RET if x.maakunta == county][0]

    if len(ba_targets) == 0:
        scale_coeffs = [[1] * max(TreeSpecies), [1] * max(TreeSpecies)]
    else:
        scale_coeffs = [[], []]
        for i in range(2):
            for target, generated in zip(ba_targets[i][0], ba_sums[i]):
                coeff = target / generated if generated > 0 else -1
                scale_coeffs[i].append(coeff)
                # print(i+1,generated, target, coeff)
    scale_coeff_ret = ba_target_ret / ba_sum_ret if ba_sum_ret > 0 else -1

    for stand in stands:
        assert stand.land_use_category is not None

        trees = stand.reference_trees
        is_retained = trees.management_category == 2
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
            strata.sapling_stratum[i_stratum] = False

            # basal area of the whole stand
            stand_ba += stratum_ba

        trees.delete(np.where(trees.stems_per_ha < 0.00005)[0])

        stand.basal_area = stand_ba
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


def area_ha_to_1000ha(stands: StandList, **operation_params):
    # Converts area of a stand from ha to 1000 ha
    _ = operation_params
    for stand in stands:
        stand.area_weight = stand.area_weight / 1000
        stand.area = stand.area / 1000
    return stands


def scale_trees_by_area_weight_factors(stands: StandList, **operation_params):
    # scale the number of stems of the (measured) trees according to the stand's
    # proportion of the sample plot. Trees with diameter in [4.5,9.5) are scaled
    # by proportion of the sample plot having 4 m radius. Trees having diameter >= 9.5 cm
    # are scaled by proportion of the sample plot with 9 m radius.
    _ = operation_params
    for stand in stands:
        trees = stand.reference_trees
        smaller_diameter = (4.5 <= trees.breast_height_diameter < 9.5) & (0 < stand.area_weight_factors[0] < 1)
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


__all__ = ['preproc_filter',
           'compute_location_metadata',
           'generate_reference_trees',
           'scale_area_weight',
           'convert_coordinates']
