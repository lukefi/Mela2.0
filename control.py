from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from lukefi.metsi.domain.pre_ops import filter_trees, generate_reference_trees, filter_stands, scale_area_weight
from lukefi.metsi.sim.generators import Alternatives, Event, Sequence
from lukefi.metsi.sim.sim_configuration import Transition
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.treatment import do_nothing
from examples.declarations.export_prepro import mela_decl


control_structure = {
    "app_configuration": {
        "state_format": "vmi13",  # options: fdm, vmi12, vmi13, xml, gpkg
        "run_modes": ["preprocess", "export_prepro", "simulate"],
        "preprocessing_output_file": "preprocessing_results",
        "simulation_output_file": "simulation_results"
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
                "mask": lambda stand: ~stand.reference_trees.sapling & (stand.reference_trees.stems_per_ha > 0)
            }
        ]
    },
    "simulation_instructions": [
        SimulationInstruction(
            events=[
                Alternatives([
                    Event(treatment=do_nothing, static_parameters={"n": 1}, tags={"first_type"}),
                    Sequence([
                        Event(treatment=do_nothing, static_parameters={"n": 2}, tags={"second_type"}, db_output=True),
                        Event(treatment=do_nothing, static_parameters={"n": 3}, tags={"third_type"}, db_output=True)
                    ])
                ])
            ]
        )
    ],
    "transition": Transition(grow_acta_fn),
    "end_condition": ForestCondition(lambda x: x.computational_unit.year >= 2050),
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
        'rst': mela_decl
    }
}

__all__ = ['control_structure']
