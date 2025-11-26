from lukefi.mela2.data.vectorize import vectorize
from lukefi.mela2.domain.natural_processes.grow_acta import grow_acta_vectorized
from lukefi.mela2.domain.pre_ops import generate_reference_trees, preproc_filter
from lukefi.mela2.sim.simulation_instruction import SimulationInstruction
from lukefi.mela2.sim.generators import Sequence, Event


control_structure = {
    "app_configuration": {
        "state_format": "vmi13",  # options: fdm, vmi12, vmi13, xml, gpkg
        "strata_origin": 2,
        "run_modes": ["preprocess", "export_prepro", "simulate"]
    },
    "preprocessing_operations": [
        generate_reference_trees,  # reference trees from strata, replaces existing reference trees
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
                "remove trees": "sapling or stems_per_ha == 0",
                "remove stands": "(site_type_category == 0) or (site_type_category == None)",  # not reference_trees
            }
        ]
    },
    "simulation_instructions": [
        SimulationInstruction(
            time_points=[2020],
            events=Sequence([
                Event(grow_acta_vectorized)
            ])
        )
    ],
    'export_prepro': {
        # "csv": {},  # default csv export
        # "rst": {},
        # "json": {}
        "pickle": {},
        "npy": {},
        "npz": {},
    }
}

__all__ = ['control_structure']
