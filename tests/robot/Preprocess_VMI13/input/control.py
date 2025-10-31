from lukefi.metsi.data.vectorize import vectorize
from lukefi.metsi.domain.pre_ops import generate_reference_trees, preproc_filter, scale_area_weight


control_structure = {
    "app_configuration": {
        "state_format": "vmi13", 
        "run_modes": ["preprocess", "export_prepro"]
    },
    "preprocessing_operations": [
        scale_area_weight,
        generate_reference_trees,
        preproc_filter,
        vectorize
    ],
    "preprocessing_params": {
        generate_reference_trees: [
            {
                "n_trees": 10,
                "method": "weibull",
                "debug": False
            }
        ],
        preproc_filter: [
            {
                "remove trees": (lambda tree: tree.sapling or tree.stems_per_ha == 0),
                "remove stands": (lambda stand: (stand.site_type_category is None) or (stand.site_type_category == 0))
            }
        ]
    },
    'export_prepro': {
        'csv': {},
        'rst': {}
    }
}

__all__ = ['control_structure']
