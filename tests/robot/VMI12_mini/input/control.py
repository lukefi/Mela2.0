from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from lukefi.metsi.domain.pre_ops import filter_stands, filter_trees, scale_area_weight
from lukefi.metsi.sim.generators import Alternatives, Event, Sequence
from lukefi.metsi.sim.transition import Transition
from lukefi.metsi.sim.instructions import SimulationInstruction
from lukefi.metsi.sim.treatment import do_nothing
from examples.declarations.export_prepro import mela_decl


control_structure = {
    "app_configuration": {
        "state_format": "vmi12",
        "measured_trees": True,
        "run_modes": ["preprocess", "export_prepro", "simulate"]
    },
    "preprocessing_operations": [
        scale_area_weight,
        filter_stands,
        filter_trees,
    ],
    "preprocessing_params": {
        filter_stands: [
            {
                "remove": (lambda stand: (stand.site_type_category is None) or (stand.site_type_category == 0))
            }
        ],
        filter_trees: [
            {
                "predicate": (lambda stand: ~((stand.reference_trees.sapling != 0) |
                                              (stand.reference_trees.stems_per_ha == 0))),
            }
        ],
    },
    "simulation_instructions": [
        SimulationInstruction(
            events=[
                Alternatives([
                    Event(treatment=do_nothing, static_parameters={"n": 1}, tags={"first_type"}),
                    Sequence([
                        Event(treatment=do_nothing, static_parameters={"n": 2}, tags={"second_type"}),
                        Event(treatment=do_nothing, static_parameters={"n": 3}, tags={"third_type"})
                    ])
                ])
            ]
        )
    ],
    "transition": Transition(grow_acta_fn, db_output=False),
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
