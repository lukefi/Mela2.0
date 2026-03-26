from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.pre_ops import (
    compute_location_metadata,
    filter_stands,
    filter_trees,
    generate_reference_trees,
    scale_area_weight)
from lukefi.metsi.domain.natural_processes.grow_motti_dll import grow_motti_dll_fn
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.generators import Alternatives, Event, Sequence
from lukefi.metsi.sim.sim_configuration import Transition
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.treatment import do_nothing

control_structure = {
    "app_configuration": {
        "state_format": "xml",
        "run_modes": ["preprocess", "simulate"],
    },
    "preprocessing_operations": [
        scale_area_weight,
        generate_reference_trees,
        compute_location_metadata,
        filter_stands,
        filter_trees
    ],
    "preprocessing_params": {
        generate_reference_trees: [
            {
                "n_trees": 10,
                "method": "weibull",
                "debug": False,
                "delete_strata": False
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
    "transition": Transition(grow_motti_dll_fn),
    "end_condition": Condition[ForestStand](lambda x: x.computational_unit.year > 2030)
}

__all__ = ['control_structure']
