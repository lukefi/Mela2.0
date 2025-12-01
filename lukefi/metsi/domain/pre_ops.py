import traceback
from typing import Any, Callable
import numpy as np
from lukefi.metsi.data.model import ReferenceTree, TreeStratum
from lukefi.metsi.data.enums.internal import LandUseCategory
from lukefi.metsi.domain.forestry_types import StandList
from lukefi.metsi.domain.utils.filter import applyfilter
from lukefi.metsi.domain.utils.opt_utils import opt_int, opt_float, opt_species, opt_storey
from lukefi.metsi.forestry.forestry_utils import find_matching_storey_stratum_for_tree
from lukefi.metsi.forestry.preprocessing import tree_generation, pre_util
from lukefi.metsi.forestry.preprocessing.coordinate_conversion import convert_location_to_ykj, CRS
from lukefi.metsi.forestry.preprocessing.age_supplementing import supplement_age_for_reference_trees
from lukefi.metsi.forestry.preprocessing.naslund import naslund_height
from lukefi.metsi.forestry.preprocessing.tree_generation_validation import create_stratum_tree_comparison_set, \
    debug_output_row_from_comparison_set, debug_output_header_row
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.data.vectorize import vectorize
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
    """Generate (N * stratum) reference trees for each stand using SoA data.

    With the new vectorized data model, measured trees and strata come from
    stand.reference_trees / stand.tree_strata instead of *_pre_vec lists.
    """

    debug = operation_params.get('debug', False)
    debug_output_rows: list[list[Any]] = []
    debug_strata_rows: list[list[Any]] = []
    debug_tree_rows: list[list[Any]] = []

    stratum_association_diameter_threshold = operation_params.get(
        'stratum_association_diameter_threshold', 2.5
    )

    for i, stand in enumerate(stands):
        print(f"\rGenerating trees for stand {stand.identifier}    {i}/{len(stands)}", end="")

        # Shortcuts to SoA containers
        strata_vec: TreeStrata = stand.tree_strata
        trees_vec: ReferenceTrees = stand.reference_trees

        # Build AoS strata objects we can hand to the existing tree_generation logic
        strata_aos: list[TreeStratum] = []
        for idx in range(strata_vec.size):
            s = TreeStratum()
            s.stand = stand
            s.identifier = str(strata_vec.identifier[idx])
            s.species = opt_species(strata_vec.species[idx])
            s.origin = opt_int(strata_vec.origin[idx])
            s.stems_per_ha = opt_float(strata_vec.stems_per_ha[idx])
            s.mean_diameter = opt_float(strata_vec.mean_diameter[idx])
            s.mean_height = opt_float(strata_vec.mean_height[idx])
            s.breast_height_age = opt_float(strata_vec.breast_height_age[idx])
            s.biological_age = opt_float(strata_vec.biological_age[idx])
            s.basal_area = opt_float(strata_vec.basal_area[idx])
            s.saw_log_volume_reduction_factor = opt_float(
                strata_vec.saw_log_volume_reduction_factor[idx]
            )
            s.cutting_year = opt_int(strata_vec.cutting_year[idx])
            s.age_when_10cm_diameter_at_breast_height = opt_int(
                strata_vec.age_when_10cm_diameter_at_breast_height[idx]
            )
            s.tree_number = opt_int(strata_vec.tree_number[idx])
            s.stand_origin_relative_position = tuple(
                float(v)
                for v in strata_vec.stand_origin_relative_position[idx]
            )
            s.lowest_living_branch_height = opt_float(
                strata_vec.lowest_living_branch_height[idx]
            )
            s.management_category = opt_int(strata_vec.management_category[idx])
            s.storey = opt_storey(strata_vec.storey[idx])
            s.sapling_stems_per_ha = opt_float(strata_vec.sapling_stems_per_ha[idx])
            s.sapling_stratum = bool(strata_vec.sapling_stratum[idx])
            s.number_of_generated_trees = opt_int(
                strata_vec.number_of_generated_trees[idx]
            )
            strata_aos.append(s)

        # Build AoS measured tree objects (used for LM tree generation / stratum matching)
        trees_aos: list[ReferenceTree] = []
        for idx in range(trees_vec.size):
            t = ReferenceTree()
            t.stand = stand
            t.identifier = str(trees_vec.identifier[idx])
            t.species = opt_species(trees_vec.species[idx])
            t.breast_height_diameter = opt_float(
                trees_vec.breast_height_diameter[idx]
            )
            # existing model & measured heights and stems are used in LM generation
            t.height = opt_float(trees_vec.height[idx])
            t.measured_height = opt_float(trees_vec.measured_height[idx])
            t.stems_per_ha = opt_float(trees_vec.stems_per_ha[idx])
            t.storey = opt_storey(trees_vec.storey[idx])
            t.tuhon_ilmiasu = str(trees_vec.tuhon_ilmiasu[idx]) or None
            trees_aos.append(t)

        # Associate measured trees to strata (same logic as earlier, just using AoS lists)
        stand_trees_sorted = sorted(
            trees_aos,
            key=lambda tree: tree.identifier if tree.identifier is not None else "",
        )

        for tree in stand_trees_sorted:
            stratum = find_matching_storey_stratum_for_tree(
                tree,
                strata_aos,
                stratum_association_diameter_threshold,
            )
            if stratum is None:
                continue

            # Attach tree to stratum for LM model (tree_generation_lm reads _trees)
            if stratum.__dict__.get("_trees") is not None:
                getattr(stratum, "_trees").append(tree)
            else:
                setattr(stratum, "_trees", [tree])

            if debug:
                debug_tree_rows.append(
                    [
                        stratum.identifier,
                        tree.breast_height_diameter or "NA",
                        tree.measured_height or "NA",
                        tree.stems_per_ha or "NA",
                    ]
                )

        # Now run the actual tree generation per stratum
        strata_aos.sort(key=lambda s: s.identifier if s.identifier is not None else "")
        new_trees: list[ReferenceTree] = []

        for stratum in strata_aos:
            stratum_trees: list[ReferenceTree] = []
            try:
                stratum_trees = tree_generation.reference_trees_from_tree_stratum(
                    stratum, **operation_params
                )
            except Exception as e:  # pylint: disable=broad-except
                print(
                    f"\nError generating trees for stratum {stratum.identifier} with "
                    f"diameter {stratum.mean_diameter}, height {stratum.mean_height}, "
                    f"basal_area {stratum.basal_area}"
                )
                print()
                if debug:
                    traceback.print_exc()
                    continue
                raise e

            stand_tree_count = len(new_trees)
            for j, tree in enumerate(stratum_trees):
                tree.identifier = f"{stand.identifier}-{stand_tree_count + j + 1}-tree"
                new_trees.append(tree)

            validation_set = create_stratum_tree_comparison_set(stratum, stratum_trees)

            if debug:
                debug_strata_rows.append(
                    [
                        stratum.identifier,
                        stratum.mean_diameter,
                        stratum.mean_height,
                        stand.basal_area,
                        stratum.basal_area,
                        stratum.species.value if stratum.species is not None else None,
                        stand.degree_days,
                    ]
                )
                debug_output_rows.append(
                    debug_output_row_from_comparison_set(stratum, validation_set)
                )

        # Vectorize the newly generated reference trees into the stand
        new_attr_dict: dict[str, list[Any]] = {}
        for t in new_trees:
            for key, value in t.__dict__.items():
                if key == "stand":
                    continue
                new_attr_dict.setdefault(key, []).append(value)

        stand.reference_trees = ReferenceTrees().vectorize(new_attr_dict)

        # Propagate number_of_generated_trees from AoS strata back into SoA
        for idx, s in enumerate(strata_aos):
            count = s.number_of_generated_trees
            if count is None:
                count = -1
            stand.tree_strata.update({"number_of_generated_trees": int(count)}, idx)

    print()

    # Debug CSV outputs
    if debug:
        import csv  # pylint: disable=import-outside-toplevel

        if len(debug_output_rows) > 0:
            with open(
                "debug_generated_tree_results.csv",
                "w",
                newline="\n",
                encoding="utf-8",
            ) as csvfile:
                writer = csv.writer(csvfile, delimiter=";")
                writer.writerow(debug_output_header_row())
                writer.writerows(debug_output_rows)

        if len(debug_strata_rows) > 1:
            with open(
                "r_strata.dat", "w", newline="\n", encoding="utf-8"
            ) as stratum_file:
                writer = csv.writer(stratum_file, delimiter=" ")
                writer.writerow(["stratum", "DGM", "HGM", "G", "Gos", "spe", "DDY"])
                writer.writerows(debug_strata_rows)

        if len(debug_tree_rows) > 1:
            with open(
                "r_trees.dat", "w", newline="\n", encoding="utf-8"
            ) as tree_file:
                writer = csv.writer(tree_file, delimiter=" ")
                writer.writerow(["stratum", "lpm", "height", "lkm"])
                writer.writerows(debug_tree_rows)

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
            # Treat NaN or <= 0 as missing
            if not (h > 0):
                d = trees.breast_height_diameter[i]
                if not (d > 0):
                    continue

                species = opt_species(trees.species[i])
                new_h = naslund_height(float(d), species)
                if new_h is None:
                    continue

                trees.update({"height": float(new_h)}, i)

    return stands


def supplement_missing_tree_ages(stands, **operation_params):
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


def supplement_missing_stratum_diameters(stands, **_):
    for stand in stands:
        if stand.tree_strata.size == 0:
            continue
        pre_util.supplement_mean_diameter(
            stand.tree_strata,
            land_use_category=stand.land_use_category,
        )
    return stands


def generate_sapling_trees_from_sapling_strata(stands, **_):
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
