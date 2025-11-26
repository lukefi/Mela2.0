from lukefi.mela2.data.model import ForestStand
from lukefi.mela2.data.vectorize import vectorize
from lukefi.mela2.domain.natural_processes.grow_acta import grow_acta
from lukefi.mela2.domain.pre_ops import generate_reference_trees, preproc_filter, scale_area_weight
from lukefi.mela2.domain.events import (
    DoNothing,
)
from lukefi.mela2.sim.condition import Condition
from lukefi.mela2.sim.generators import Alternatives, Sequence
from lukefi.mela2.sim.sim_configuration import Transition
from lukefi.mela2.sim.simulation_instruction import SimulationInstruction
from lukefi.mela2.sim.operations import do_nothing


control_structure = {
    "app_configuration": {
        "state_format": "vmi13",  # options: fdm, vmi12, vmi13, xml, gpkg
        # "state_input_container": "csv",  # Only relevant with fdm state_format. Options: pickle, json
        # "state_output_container": "csv",  # options: pickle, json, csv, null
        # "derived_data_output_container": "pickle",  # options: pickle, json, null
        "run_modes": ["preprocess", "export_prepro", "simulate"]
    },
    "preprocessing_operations": [
        scale_area_weight,
        generate_reference_trees,  # reference trees from strata, replaces existing reference trees
        preproc_filter,
        vectorize
        # "supplement_missing_tree_heights",
        # "supplement_missing_tree_ages",
        # "generate_sapling_trees_from_sapling_strata"
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
                # not reference_trees
                "remove stands": (lambda stand: (stand.site_type_category is None) or (stand.site_type_category == 0))
            }
        ]
    },
    "simulation_instructions": [
        SimulationInstruction(
            events=[
                Alternatives([
                    DoNothing(parameters={"n": 1}),
                    Sequence([
                        DoNothing(parameters={"n": 2}),
                        DoNothing(parameters={"n": 3})
                    ])
                ])
            ]
        )
    ],
    "transition": Transition(grow_acta),
    "end_condition": Condition[ForestStand](lambda x: x.year >= 2050),
    "post_processing": {
        "operation_params": {
            do_nothing: [
                {"param": "value"}
            ]
        },
        "post_processing": [
            do_nothing
        ]
    },
    'export_prepro': {
        'csv': {},
        'json': {}
    }
}

__all__ = ['control_structure']
