from lukefi.metsi.domain.conditions import TimePoints
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta
from lukefi.metsi.domain.pre_ops import generate_reference_trees, preproc_filter
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.generators import Sequence, Event


control_structure = {
    "app_configuration": {
        "state_format": "vmi13",  # options: fdm, vmi12, vmi13, xml, gpkg
        "strata_origin": 2,
        "run_modes": ["preprocess", "export_prepro", "simulate"]
    },
    "preprocessing_operations": [
        generate_reference_trees,  # reference trees from strata, replaces existing reference trees
        preproc_filter,
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
                "remove trees": (lambda trees: (trees.sapling != 0) | (trees.stems_per_ha == 0)),
                "remove stands": (lambda stand: (stand.site_type_category is None) or (stand.site_type_category == 0))
            }
        ]
    },
    "simulation_instructions": [
        SimulationInstruction(
            conditions=[
                TimePoints([2020])
            ],
            events=Sequence([
                Event(grow_acta)
            ])
        )
    ],
    'export_prepro': {
        "csv": {},
    }
}

__all__ = ['control_structure']
