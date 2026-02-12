from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from lukefi.metsi.domain.pre_ops import generate_reference_trees, preproc_filter
from lukefi.metsi.sim.generators import Alternatives, Event
from lukefi.metsi.sim.sim_configuration import Transition
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.treatment import do_nothing
from examples.declarations.export_prepro import mela_and_default_csv
from user_events import Harvest20percent


control_structure = {
    "app_configuration": {
        "state_format": "vmi13",
        "run_modes": ["preprocess", "export_prepro", "simulate"]
    },
    "preprocessing_operations": [
        generate_reference_trees,
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
            events=[
                Alternatives([
                    Event(treatment=do_nothing, tags={"do_nothing"}),
                    Harvest20percent()
                ])
            ]
        )
    ],
    "transition": Transition(grow_acta_fn),
     # 30-year simulation
    "end_condition": ForestCondition(lambda x: x.computational_unit.relative_time > 30),
    # outputs both preprocinssing result formats .csv and .rst
    'export_prepro': mela_and_default_csv
}

__all__ = ['control_structure']
