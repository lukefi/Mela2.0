from lukefi.metsi.app.metsi_control import AppConfiguration, MetsiControl
from lukefi.metsi.app.metsi_enum import RunMode, StateFormat, StrataOrigin
from lukefi.metsi.core.generators import Alternatives, Event, Sequence
from lukefi.metsi.core.instructions import SimulationInstruction
from lukefi.metsi.core.sim_control import Preprocessing, Simulation
from lukefi.metsi.core.transition import Transition
from lukefi.metsi.core.treatment import do_nothing
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.collected_data import NaturalProcessInfo
from lukefi.metsi.domain.conditions import RelativeTimePoints
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from lukefi.metsi.domain.pre_ops import filter_stands, filter_trees, generate_reference_trees, scale_area_weight

from examples.declarations.export_prepro import mela_decl
from examples.declarations.sqlite import sqlite_decl

control_structure = MetsiControl[ForestStand](
    app_configuration=AppConfiguration(
        state_format=StateFormat.VMI13,
        run_modes=[
            RunMode.PREPROCESS,
            RunMode.EXPORT_PREPRO,
            RunMode.SIMULATE
        ],
        preprocessing_output_file="preprocessing_results",
        simulation_output_file="simulation_results",
        sqlite_decl=sqlite_decl,
        measured_trees=False,
        strata=True,
        strata_origin=StrataOrigin.INVENTORY
    ),
    preprocessing=Preprocessing(
        operations=[
            scale_area_weight,
            generate_reference_trees,
            filter_stands,
            filter_trees
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
        },
    ),
    export_prepro={
        "csv": {},
        "csv_exp": {},
        "rst": mela_decl
    },
    simulation=Simulation(
        instructions=[
            SimulationInstruction(
                conditions=[
                    RelativeTimePoints([1, 3, 4, 5])
                ],
                events=[
                    Alternatives(
                        [
                            Event(treatment=do_nothing, static_parameters={"n": 1}, tags={"first_type"}),
                            Sequence(
                                [
                                    Event(treatment=do_nothing, static_parameters={"n": 2}, tags={"second_type"}, db_output=True),
                                    Event(treatment=do_nothing, static_parameters={"n": 3}, tags={"third_type"}, db_output=True)
                                ]
                            )
                        ]
                    )
                ]
            )
        ],
        transition=Transition(grow_acta_fn, 50, {NaturalProcessInfo}),
        end_condition=ForestCondition(lambda x: x.unit.year >= 2050),
    ),
)
