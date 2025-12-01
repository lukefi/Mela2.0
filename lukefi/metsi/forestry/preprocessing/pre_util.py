from typing import Any
import numpy as np
from lukefi.metsi.data.vector_model import TreeStrata
from lukefi.metsi.data.enums.internal import LandUseCategory

DIAMETER_FACTOR = 1.2


def supplement_mean_diameter(strata: TreeStrata,
                             _land_use_category: LandUseCategory | None = None) -> None:
    """
    Mutates strata.mean_diameter in place when diameter is missing but height is present
    and the domain rules say we should supplement (scrub land, sapling strata etc.).
    """
    # missing or non-positive diameter
    md = strata.mean_diameter
    mh = strata.mean_height

    # mask “height > 0 and diameter is nan or <= 0”
    md_missing = np.logical_or(np.isnan(md), md <= 0.0)
    mh_valid = mh > 0.0

    # optional additional rules – mirror your earlier pre_ops logic here
    mask = np.logical_and(md_missing, mh_valid)

    # if you have land_use_category-based rules, you can add them here:
    # if land_use_category == LandUseCategory.SCRUB_LAND:
    #     mask = np.logical_and(mask, mh > 1.3)

    strata.mean_diameter[mask] = mh[mask] * DIAMETER_FACTOR


def create_sapling_rows_from_strata(strata: TreeStrata, stand_identifier: str,
                                    base_tree_number: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(strata.size):
        if not bool(strata.sapling_stratum[i]):
            continue

        tree_number = base_tree_number + len(rows) + 1
        rows.append(
            {
                "identifier": f"{stand_identifier}-{tree_number}-tree",
                "tree_number": tree_number,
                "stems_per_ha": strata.sapling_stems_per_ha[i],
                "species": strata.species[i],
                "breast_height_diameter": strata.mean_diameter[i],
                "height": strata.mean_height[i],
                "breast_height_age": strata.breast_height_age[i],
                "biological_age": strata.biological_age[i],
                "saw_log_volume_reduction_factor": -1.0,
                "pruning_year": 0,
                "age_when_10cm_diameter_at_breast_height": 0,
                "origin": strata.origin[i],
                "stand_origin_relative_position": (0.0, 0.0, 0.0),
                "management_category": 1,
                "sapling": True,
            }
        )
    return rows
