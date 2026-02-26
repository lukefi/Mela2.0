import numpy as np

from lukefi.metsi.domain.pre_ops import *
from lukefi.metsi.sim.generators import *

control_structure = {
    "app_configuration": {
        "state_format": "vmi13",
        "measured_trees": True,
        "run_modes": ["preprocess", "export_prepro"]
    },

    "preprocessing_operations": [
        filter_stands,
        filter_trees,
        scale_area_weight,
        generate_reference_trees
    ],

    "preprocessing_params": {
        filter_stands: [
            {
                "remove": lambda stand: stand.site_type_category is None
            }
        ],
        filter_trees: [
            {
                "predicate": lambda stand: np.isin(stand.reference_trees.tree_type, ('V', 'U', 'S', 'T', 'N'))
            }
        ],
        generate_reference_trees: [
            {
                "n_trees": 10,
                "method": "lm",  # lm, weibull
                "stratum_association_diameter_threshold": 2.5,
                "lm_mode": "dcons",  # dcons, fcons
                "lm_shdef": 5,
                "debug": True  # true, false
            }
        ]
    },
    "export_prepro": {
        "csv": {}  # default csv export
    }
}

__all__ = ["control_structure"]
