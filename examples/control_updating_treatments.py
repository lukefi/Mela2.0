from lukefi.metsi.domain.collected_data import NaturalProcessInfo
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from lukefi.metsi.domain.pre_ops import filter_trees, generate_reference_trees, filter_stands, scale_area_weight
from lukefi.metsi.sim.instructions import UpdatingInstructions
from examples.declarations.export_prepro import mela_decl
from examples.declarations.sqlite import sqlite_decl
from lukefi.metsi.sim.transition import Transition


control_structure = {
    "app_configuration": {
        "state_format": "xml",  # options: fdm, vmi12, vmi13, xml, gpkg
        "run_modes": ["preprocess", "update", "export_prepro"],
        "preprocessing_output_file": "preprocessing_results",
        "simulation_output_file": "simulation_results",
        "sqlite_decl": sqlite_decl,
    },
    "preprocessing_operations": [
        scale_area_weight,
        generate_reference_trees,  # reference trees from strata, replaces existing reference trees
        filter_stands,
        filter_trees,
    ],
    "preprocessing_params": {
        generate_reference_trees: [
            {
                "n_trees": 10,
                "method": "weibull",
                "debug": False
            }
        ],
        filter_stands: [
            {
                "remove": (lambda stand: (stand.site_type_category is None) or (stand.site_type_category == 0))
            }
        ],
        filter_trees: [
            {
                "predicate": lambda stand: ~stand.reference_trees.sapling & (stand.reference_trees.stems_per_ha > 0)
            }
        ]
    },
    "updating": UpdatingInstructions(
        target_time=2026,
        transition=Transition(grow_acta_fn,
                              db_output_state=True,
                              db_output_cd=True,
                              collected_data={NaturalProcessInfo}),
        output_treatment_state=True,
        output_treatment_cd=True
    ),
    'export_prepro': {
        'csv': {},
        'csv_exp': {},
        'rst': mela_decl
    }
}

__all__ = ['control_structure']
