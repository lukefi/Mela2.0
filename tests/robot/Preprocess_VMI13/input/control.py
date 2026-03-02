from lukefi.metsi.domain.pre_ops import filter_stands, filter_trees, generate_reference_trees, scale_area_weight


control_structure = {
    "app_configuration": {
        "state_format": "vmi13",
        "run_modes": ["preprocess", "export_prepro"]
    },
    "preprocessing_operations": [
        scale_area_weight,
        generate_reference_trees,
        filter_stands,
        filter_trees
    ],
    "preprocessing_params": {
        generate_reference_trees: [
            {
                "n_trees": 10,
                "method": "weibull",
                "debug": False,
                "delete_strata": True
            }
        ],
        filter_stands: [
            {
                "remove": (lambda stand: (stand.site_type_category is None) or (stand.site_type_category == 0))
            }
        ],
        filter_trees: [
            {
                "predicate": (lambda stand: ~((stand.reference_trees.sapling != 0) |
                                              (stand.reference_trees.stems_per_ha == 0)))
            }
        ]
    },
    'export_prepro': {
        'csv': {},
        'rst': {}
    }
}

__all__ = ['control_structure']
