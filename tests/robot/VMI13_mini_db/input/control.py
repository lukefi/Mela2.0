from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.conditions import TimePoints
from lukefi.metsi.domain.events import GrowActa, GrowMetsi
from lukefi.metsi.domain.pre_ops import filter_stands, filter_trees, generate_reference_trees, scale_area_weight
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.generators import Alternatives
from lukefi.metsi.sim.operations import do_nothing
from lukefi.metsi.sim.sim_configuration import Transition
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction


control_structure = {
    "app_configuration": {
        "state_format": "vmi13",
        "run_modes": ["preprocess", "simulate"]
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
    "simulation_instructions": [
        SimulationInstruction(
            conditions=[TimePoints([2018, 2023, 2028])],
            events=[
                Alternatives([
                    GrowActa(),
                    GrowMetsi(),
                ])
            ]
        )
    ],
    "transition": Transition(do_nothing, db_output=False),
    "end_condition": Condition[ForestStand](lambda x: x.computational_unit.time > 2028)
}

__all__ = ['control_structure']
