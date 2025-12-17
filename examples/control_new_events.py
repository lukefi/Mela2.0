from lukefi.metsi.data.vectorize import vectorize
from lukefi.metsi.domain.conditions import TimePoints
from lukefi.metsi.domain.pre_ops import generate_reference_trees
from lukefi.metsi.domain.events import GrowMetsi
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction

from user_events import Mounding

control_structure = {
    "app_configuration": {
        "state_format": "xml",
        "formation_strategy": "partial",
        "evaluation_strategy": "depth",
        "run_modes": ["preprocess", "simulate"],
        "state_output_container": "csv",
    },
    "preprocessing_operations": [
        generate_reference_trees,
        vectorize,
    ],
    "preprocessing_params": {
        generate_reference_trees: [
            {
                "n_trees": 10,
                "method": "weibull",
                "debug": False
            }
        ]
    },
    "simulation_instructions": [

        SimulationInstruction(
            conditions=[
                TimePoints([2020, 2025, 2030, 2035, 2040, 2045, 2050])
            ],
            events=[
                GrowMetsi()
            ]
        ),

        SimulationInstruction(
            conditions=[
                TimePoints([2020])
            ],
            events=[Mounding()]
        ),
    ],
}


__all__ = ['control_structure']
