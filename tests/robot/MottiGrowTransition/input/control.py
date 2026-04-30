from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.collected_data import NaturalProcessInfo
from lukefi.metsi.domain.conditions import TimePoints
from lukefi.metsi.domain.natural_processes.grow_motti_dll import grow_motti_dll_fn
from lukefi.metsi.domain.pre_ops import (
    compute_location_metadata,
    filter_stands,
    filter_trees,
    generate_reference_trees,
    scale_area_weight)
from lukefi.metsi.domain.events import DoNothing
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.sim_configuration import Transition
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.domain.natural_processes.motti_bootstrap import initialize_motti


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
            conditions=[TimePoints([2025, 2030, 2035])],
            events=[
                DoNothing()
            ]
        )
    ],
    "transition": Transition[ForestStand](grow_motti_dll_fn,
                                          max_step=5,
                                          collected_data={NaturalProcessInfo},
                                          name="grow_motti",
                                          db_output=True,
                                          db_output_state=True,
                                          db_output_cd=True,
                                          init_fn=initialize_motti),
    "end_condition": Condition[ForestStand](lambda x: x.computational_unit.year > 2030)
}

__all__ = ['control_structure']
