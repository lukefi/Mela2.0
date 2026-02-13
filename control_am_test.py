import numpy as np
from lukefi.metsi.data.vectorize import vectorize
from lukefi.metsi.domain.pre_ops import (
    area_ha_to_1000ha,
    convert_coordinates,
    filter_trees,
    generate_reference_trees,
    filter_stands,
    scale_basal_area_at_county_level,
    scale_trees_by_area_weight_factors,
    update_strata_to_match_trees)

control_structure = {
    "app_configuration": {
        "state_format": "vmi13",
        "strata": True,
        "measured_trees": True,
        "run_modes": ["preprocess", "export_prepro"]
    },
    "preprocessing_operations": [
        vectorize,
        filter_stands,
        filter_trees,
        scale_trees_by_area_weight_factors,
        generate_reference_trees,
        scale_basal_area_at_county_level,
        update_strata_to_match_trees,
        area_ha_to_1000ha,
        convert_coordinates
    ],
    "preprocessing_params": {
        filter_stands: [
            {
                "select": (lambda stand: stand.land_use_category in (1, 2, 3) and not stand.auxiliary_stand)
            }
        ],
        filter_trees: [
            {
                "mask": (
                    lambda stand: np.isin(
                        stand.reference_trees.tree_type,
                        ("",
                         "V",
                         "U",
                         "S",
                         "T",
                         "N")) & np.isin(
                        stand.reference_trees.tree_category,
                        ("",
                         "0",
                         "1",
                         "3",
                         "7")))
            }
        ],
        generate_reference_trees: [
            {
                "n_trees": 10,
                "stratum_association_diameter_threshold": 2.5,
                "ng_scale_factor": 1.0,
                "method": "lm",
                "lm_n_trees_mode": "calc",  # param (use n_trees), calc (number of generated trees depends on g)
                "lm_n_trees": 15,  # (about) max number of generated trees. if not set n_trees is used as defaul
                "lm_gos_div": 0.5,  # number of generated trees ~ stratum_g/gos_div
                "lm_stems_mode": "lkm0",  # lkm (restricted), lkm0 (unrestricted)
                "lm_stems_nmax": 20000,
                "lm_mode": "fcons",  # dcons, fcons, fixw
                "lm_fix_width": 2,  # used with lm_mode fixw
                "lm_shdef": 5,
                "age_model": True,  # käytetäänkö kuvauspuulle ikämallin ikää (true) vai ositteen ikää (false)
            }
        ],
        convert_coordinates: [
            {
                "target_system": "YKJ"
            }
        ]
    },
    'export_prepro': {
        'csv': {},
        'rst': {}}}

__all__ = ['control_structure']
