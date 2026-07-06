from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.collected_data import NaturalProcessInfo
from lukefi.metsi.domain.conditions import RelativeTimePoints
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.updating import update_to_year
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from lukefi.metsi.domain.pre_ops import filter_trees, generate_reference_trees, filter_stands, scale_area_weight
from lukefi.metsi.app.metsi_control import AppConfiguration, MetsiControl
from lukefi.metsi.sim.generators import Alternatives, Event, Sequence
from lukefi.metsi.sim.sim_control import Preprocessing, Simulation
from lukefi.metsi.sim.transition import Transition
from lukefi.metsi.sim.instructions import SimulationInstruction
from lukefi.metsi.sim.treatment import do_nothing
from examples.declarations.export_prepro import mela_decl
from examples.declarations.sqlite import sqlite_decl


control_structure = MetsiControl[ForestStand](
    app_configuration=AppConfiguration(
        state_format=StateFormat.VMI13,  # options: fdm, vmi12, vmi13, xml, gpkg
        run_modes=[RunMode.PREPROCESS, RunMode.EXPORT_PREPRO, RunMode.SIMULATE],
        preprocessing_output_file="preprocessing_results",
        simulation_output_file="simulation_results",
        sqlite_decl=sqlite_decl,
    ),
    preprocessing=Preprocessing[ForestStand](
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
        }),
    simulation=Simulation[ForestStand](
        instructions=[
            SimulationInstruction(
                conditions=[
                    RelativeTimePoints([0])
                ],
                events=[
                    Event(treatment=update_to_year, static_parameters={"transition": grow_acta_fn,
                                                                       "target_year": 2020})
                ]
            ),
            SimulationInstruction(
                conditions=[
                    RelativeTimePoints([1, 2, 3, 4, 5, 10, 15, 20, 25, 30])
                ],
                events=[
                    Alternatives([
                        Event(treatment=do_nothing, static_parameters={"n": 1}, tags={"first_type"}),
                        Sequence([
                            Event(
                                treatment=do_nothing,
                                static_parameters={
                                    "n": 2},
                                tags={"second_type"},
                                db_output=True),
                            Event(treatment=do_nothing, static_parameters={"n": 3}, tags={"third_type"}, db_output=True)
                        ])
                    ])
                ]
            )
        ],
        transition=Transition(grow_acta_fn, 5, collected_data={NaturalProcessInfo}, name="grow_acta"),
        end_condition=ForestCondition(lambda x: x.unit.year >= 2050)),
    export_prepro={
        'csv': {},
        'csv_exp': {},
        'rst': mela_decl
    }
)

__all__ = ['control_structure']
