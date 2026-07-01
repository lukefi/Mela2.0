from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from lukefi.metsi.domain.pre_ops import filter_trees, generate_reference_trees, filter_stands, scale_area_weight
from lukefi.metsi.sim.instructions import UpdatingInstructions
from examples.declarations.export_prepro import mela_decl
from examples.declarations.sqlite import sqlite_decl
from lukefi.metsi.sim.sim_control import AppConfiguration, MetsiControl, Preprocessing


control_structure = MetsiControl[ForestStand](
    app_configuration=AppConfiguration(
        state_format=StateFormat.XML,  # options: fdm, vmi12, vmi13, xml, gpkg
        run_modes=[RunMode.PREPROCESS, RunMode.UPDATE, RunMode.EXPORT_PREPRO],
        preprocessing_output_file="preprocessing_results",
        simulation_output_file="simulation_results",
        sqlite_decl=sqlite_decl,
    ),
    preprocessing=Preprocessing(
        operations=[
            scale_area_weight,
            generate_reference_trees,  # reference trees from strata, replaces existing reference trees
            filter_stands,
            filter_trees,
        ],
        params={
            generate_reference_trees: [
                {
                    "n_trees": 10,
                    "method": "weibull",
                    "debug": False
                }
            ],
            filter_stands: [
                {
                    "remove": (lambda stand: (stand.site_type_category is None) or (stand.site_type_category == 0))
                }
            ],
            filter_trees: [
                {
                    "predicate": lambda stand: ~stand.reference_trees.sapling & (stand.reference_trees.stems_per_ha > 0)
                }
            ]
        }
    ),
    updating=UpdatingInstructions(
        2026,
        grow_acta_fn,
        output_transition_state=True,
        output_transition_cd=True,
        output_treatment_state=True,
        output_treatment_cd=True
    ),
    export_prepro={
        'csv': {},
        'csv_exp': {},
        'rst': mela_decl
    }
)

__all__ = ['control_structure']
