import numpy as np
from lukefi.metsi.domain.conditions import MinimumTimeInterval
from lukefi.metsi.domain.forestry_operations.ftrt_soil_surface_preparation import ftrt_soil_surface_preparation

from lukefi.metsi.data.vectorize import vectorize
from lukefi.metsi.domain.pre_ops import (
    compute_location_metadata,
    generate_reference_trees,
    preproc_filter,
    scale_area_weight)
from lukefi.metsi.domain.events import (
    GrowMetsi,
    SoilSurfacePreparation,
    )
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.generators import Alternatives
from lukefi.metsi.sim.generators import Sequence

control_structure = {
    "app_configuration": {
        "state_format": "xml",  # options: fdm, vmi12, vmi13, xml, gpkg
        # "state_input_container": "csv",  # Only relevant with fdm state_format. Options: pickle, json
        # "state_output_container": "csv",  # options: pickle, json, csv, null
        # "derived_data_output_container": "pickle",  # options: pickle, json, null
        "formation_strategy": "partial",
        "evaluation_strategy": "depth",
        #        "run_modes": ["preprocess", "export_prepro", "simulate", "postprocess", "export"]
        "run_modes": ["preprocess", "simulate"],
        "state_output_container": "csv",
    },
    "preprocessing_operations": [
        scale_area_weight,
        generate_reference_trees,  # reference trees from strata, replaces existing reference trees
        compute_location_metadata,
        preproc_filter,
        vectorize,
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
                "remove trees": "sapling or stems_per_ha == 0",
                "remove stands": "(site_type_category == None) or (site_type_category == 0)",  # not reference_trees
            }
        ]
    },
    "simulation_instructions": [


        SimulationInstruction(
            time_points=[2020, 2025, 2030, 2035, 2040, 2045, 2050],
            events=[
                GrowMetsi()
            ]
        ),

        # Soil surface preparation (metadata)
        SimulationInstruction(
            time_points=[2025],
            events=[SoilSurfacePreparation(parameters={
                "method": "mounding",
                "intensity": 1200.0,
                "labels": ["ssp_default"]
            })],
            conditions=[MinimumTimeInterval(20, ftrt_soil_surface_preparation)]
        ),

    ],
}


__all__ = ['control_structure']