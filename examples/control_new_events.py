from lukefi.metsi.data.vectorize import vectorize
from lukefi.metsi.domain.pre_ops import generate_reference_trees
from lukefi.metsi.domain.events import (
    GrowMetsi,
    SoilSurfacePreparation,
    )
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction

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
            time_points=[2020, 2025, 2030, 2035, 2040, 2045, 2050],
            events=[
                GrowMetsi(),
                SoilSurfacePreparation()
            ]
        ),

        # Soil surface preparation (metadata)
        SimulationInstruction(
            time_points=[2020],
            events=[SoilSurfacePreparation(),
            ]
        ),
    ],
}


__all__ = ['control_structure']
