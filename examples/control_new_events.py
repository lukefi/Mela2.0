import numpy as np
from lukefi.metsi.domain.conditions import MinimumTimeInterval
from lukefi.metsi.domain.forestry_operations.ftrt_cutting import ftrt_cutting
from lukefi.metsi.domain.forestry_operations.ftrt_soil_surface_preparation import ftrt_soil_surface_preparation
from lukefi.metsi.domain.forestry_operations.ftrt_mark_trees import ftrt_mark_trees

from lukefi.metsi.domain.forestry_operations.events_regeneration import event_regeneration_chain
from lukefi.metsi.domain.forestry_operations.events_r_ported import (
    event_first_thinning_ajourat,
    event_ba_thinning_from_below_regular_coniferPriority,
    event_ba_thinning_from_below_regular_coniferPriority2,
    event_ba_thinning_from_below_regular_coniferPriority3,
    event_ba_thinning_from_below_regular_coniferBirchPriority,
)

# --- Optional: thinning cores if you want wrappers for them too
from lukefi.metsi.domain.forestry_operations.thin_basal_area import ftrt_thin_basal_area
from lukefi.metsi.domain.forestry_operations.thin_nr_of_stems import ftrt_thin_nr_of_stems

from lukefi.metsi.data.vectorize import vectorize
from lukefi.metsi.domain.pre_ops import (
    compute_location_metadata,
    generate_reference_trees,
    preproc_filter,
    scale_area_weight)
from lukefi.metsi.domain.events import (
    DoNothing,
    GrowMotti, 
    GrowMetsi,   
    Cutting,
    SoilSurfacePreparation,
    Regeneration,
    MarkTrees,
    ThinBasalArea,
    ThinNumberOfStems,
    RegenerationChain,
    FirstThinningAjourat,
    BAThinningConiferPriority,
    BAThinningConiferPriority2,
    BAThinningConiferPriority3,
    BAThinningBirchPriority
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
        "strata_origin": 2,
        "formation_strategy": "partial",
        "evaluation_strategy": "depth",
        #        "run_modes": ["preprocess", "export_prepro", "simulate", "postprocess", "export"]
        "run_modes": ["preprocess", "export_prepro", "simulate"],
        #"state_output_container": "rst"
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

        # 1) Light generic cutting (20% stems from-below, all species)
        SimulationInstruction(
            time_points=[2025],
            events=[Cutting(parameters={
                "tree_selection": {
                    "Target": {"type": "relative", "var": "f", "amount": 0.20},
                    "sets": [{
                        "sfunction": lambda ctx, data: np.ones(data.size, dtype=bool),
                        "order_var": "d",            # diameter
                        "target_var": "f",           # stems/ha
                        "target_type": "relative",
                        "target_amount": 1.0,        # consume global target
                        "profile_x": np.array([0.0, 1.0]),
                        "profile_y": np.array([1.0, 0.0]),  # from-below tilt
                        "profile_xmode": "relative",
                    }],
                },
                "labels": ["cutting_generic"]
            })],
            conditions=[MinimumTimeInterval(10, ftrt_cutting)]
        ),

        # 2) Soil surface preparation (metadata)
        SimulationInstruction(
            time_points=[2026],
            events=[SoilSurfacePreparation(parameters={
                "method": "mounding",
                "intensity": 1200.0,
                "labels": ["ssp_default"]
            })],
            conditions=[MinimumTimeInterval(20, ftrt_soil_surface_preparation)]
        ),

        # 3) Regeneration (planting) – spruce 1800/ha, Hgm ~0.7 m, age 3
        SimulationInstruction(
            time_points=[2027],
            events=[Regeneration(parameters={
                "origin": 1,             # planted
                "species": 2,            # spruce
                "f": 1800.0,             # stems/ha
                "Hgm": 0.7,
                "Dgm": 0.0,
                "age_biol": 3.0,
                "labels": ["planting_default"]
            })]
        ),

        # 4) Mark retention trees (5% of stems, pick largest diameters)
        SimulationInstruction(
            time_points=[2028],
            events=[MarkTrees(parameters={
                "tree_selection": {
                    "Target": {"type": "relative", "var": "f", "amount": 0.05},
                    "sets": [{
                        "sfunction": lambda ctx, data: np.ones(data.size, dtype=bool),
                        "order_var": "d",
                        "target_var": "f",
                        "target_type": "relative",
                        "target_amount": 1.0,
                        "profile_x": np.array([0.0, 1.0]),
                        "profile_y": np.array([0.0, 1.0]),  # bias to largest trees
                        "profile_xmode": "relative",
                    }],
                },
                "attributes": {
                    "tree_type": "retained",
                    "management_category": 2
                },
                "labels": ["retention_marking"]
            })],
            conditions=[MinimumTimeInterval(30, ftrt_mark_trees)]
        ),

        # 5) First thinning incl. Ajourat (strip roads 18% + stems cap)
        SimulationInstruction(
            time_points=[2030],
            events=[FirstThinningAjourat() 
            ],
            conditions=[MinimumTimeInterval(15, event_first_thinning_ajourat)]
           
        ),

        # 6) Thinning by number of stems (generic core): keep >=1500, cap 50%
        SimulationInstruction(
            time_points=[2032],
            events=[ThinNumberOfStems(parameters={
                "max_proportion": 0.50,
                "stems_after": 1500.0,  # safe default; tables used in ajourat flow elsewhere
                "labels": ["first_thinning_core"]
            })],
            conditions=[MinimumTimeInterval(12, ftrt_thin_nr_of_stems)]
        ),

        # 7) Basal-area thinning (core). No params => derives target from BA tables.
        SimulationInstruction(
            time_points=[2035],
            events=[ThinBasalArea(parameters={
                # Optional: cap overall removal if you want
                # "max_proportion": 0.35,
                "labels": ["ba_thinning_core"]
            })
            ],
            conditions=[MinimumTimeInterval(12, ftrt_thin_basal_area)]
        ),

        # 8) BA thinning variants (conifer priority family)
        SimulationInstruction(
            time_points=[2040],
            events=[BAThinningConiferPriority()],
            conditions=[MinimumTimeInterval(12, event_ba_thinning_from_below_regular_coniferPriority)]
            
        ),
        SimulationInstruction(
            time_points=[2042],
            events=[BAThinningConiferPriority2()],   # keep ~2 m²/ha other species
            conditions=[MinimumTimeInterval(12, event_ba_thinning_from_below_regular_coniferPriority2)]
            
        ),
        SimulationInstruction(
            time_points=[2044],
            events=[BAThinningConiferPriority3()],   # explicit other-species cap (e.g., 0.7)
            conditions=[MinimumTimeInterval(12, event_ba_thinning_from_below_regular_coniferPriority3)]
            
        ),
        SimulationInstruction(
            time_points=[2046],
            events=[BAThinningBirchPriority()],      # flip priority to birches first
            conditions=[MinimumTimeInterval(12, event_ba_thinning_from_below_regular_coniferBirchPriority)]
            
        ),

        # 9) Full regeneration chain: retention -> clearcut -> soil prep -> planting
        SimulationInstruction(
            time_points=[2050],
            events=[RegenerationChain(parameters={
                # Retention marking (3% largest crowns)
                "par_trt_rt": {
                    "tree_selection": {
                        "Target": {"type": "relative", "var": "f", "amount": 0.03},
                        "sets": [{
                            "sfunction": lambda ctx, data: np.ones(data.size, dtype=bool),
                            "order_var": "d",
                            "target_var": "f",
                            "target_type": "relative",
                            "target_amount": 1.0,
                            "profile_x": np.array([0.0, 1.0]),
                            "profile_y": np.array([0.2, 1.0]),
                            "profile_xmode": "relative",
                        }],
                    },
                    "attributes": {"tree_type": "retained", "management_category": 2},
                    "labels": ["retention"]
                },
                # Clearcut selection (remove almost all stems)
                "par_trt_cc": {
                    "tree_selection": {
                        "Target": {"type": "relative", "var": "f", "amount": 0.995},
                        "sets": [{
                            "sfunction": lambda ctx, data: np.ones(data.size, dtype=bool),
                            "order_var": "d",
                            "target_var": "f",
                            "target_type": "relative",
                            "target_amount": 1.0,
                            "profile_x": np.array([0.0, 1.0]),
                            "profile_y": np.array([0.75, 0.90]),  # take broadly across sizes
                            "profile_xmode": "relative",
                        }],
                    },
                    "labels": ["clearcut"]
                },
                # Soil surface preparation
                "par_trt_ss": {"method": "mounding", "intensity": 1200.0, "labels": ["ssp"]},
                # Planting (spruce 1800/ha)
                "par_trt_pl": {"origin": 1, "species": 2, "f": 1800.0, "Hgm": 0.7, "Dgm": 0.0, "age_biol": 3.0,
                               "labels": ["planting"]},
            })],
            conditions=[MinimumTimeInterval(30, event_regeneration_chain)]
        ),

    ],
    "post_processing": {
        "operation_params": {
            "do_nothing": [
                {"param": "value"}
            ]
        },
        "post_processing": [
            "do_nothing"
        ]
    },
    'export_prepro': {
        # "csv": {},  # default csv export
        # "rst": {},
        # "json": {}
        "pickle": {},
        "npy": {},
        "npz": {},
    }
}


__all__ = ['control_structure']
