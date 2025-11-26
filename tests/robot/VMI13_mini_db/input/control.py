from lukefi.mela2.data.model import ForestStand
from lukefi.mela2.data.vectorize import vectorize
from lukefi.mela2.domain.conditions import TimePoints
from lukefi.mela2.domain.events import GrowActa, GrowMetsi
from lukefi.mela2.domain.pre_ops import generate_reference_trees, preproc_filter, scale_area_weight
from lukefi.mela2.sim.condition import Condition
from lukefi.mela2.sim.generators import Alternatives
from lukefi.mela2.sim.operations import do_nothing
from lukefi.mela2.sim.sim_configuration import Transition
from lukefi.mela2.sim.simulation_instruction import SimulationInstruction


control_structure = {
    "app_configuration": {
        "state_format": "vmi13",
        "formation_strategy": "partial",
        "evaluation_strategy": "depth",
        "run_modes": ["preprocess", "simulate"]
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
                "remove stands": (lambda stand: (stand.site_type_category is None) or (stand.site_type_category == 0)),
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
    "transition": Transition(do_nothing),
    "end_condition": Condition[ForestStand](lambda x: x.time > 2028)
}

__all__ = ['control_structure']
