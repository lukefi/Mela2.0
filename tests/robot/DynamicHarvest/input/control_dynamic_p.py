from user_events import Harvest20percent, FirstThinningMineralSoils

from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.core.sim_control import Preprocessing, Simulation
from lukefi.metsi.core.treatment import do_nothing
from lukefi.metsi.core.instructions import SimulationInstruction
from lukefi.metsi.core.transition import Transition
from lukefi.metsi.core.generators import Alternatives, Event, Sequence
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from lukefi.metsi.domain.pre_ops import filter_stands, filter_trees, generate_reference_trees, scale_area_weight
from lukefi.metsi.app.metsi_control import AppConfiguration, MetsiControl

from examples.declarations.export_prepro import mela_and_default_csv
from examples.declarations.sqlite import sqlite_decl

control_structure = MetsiControl[ForestStand](
    app_configuration=AppConfiguration(
        state_format=StateFormat.VMI13,
        run_modes=[RunMode.PREPROCESS, RunMode.EXPORT_PREPRO, RunMode.SIMULATE],
        sqlite_decl=sqlite_decl
    ),
    preprocessing=Preprocessing[ForestStand](
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
                                                  (stand.reference_trees.stems_per_ha == 0))),
                }
            ]

        }
    ),
    export_prepro=mela_and_default_csv,
    simulation=Simulation[ForestStand](
        instructions=[
            SimulationInstruction(
                events=[
                    Alternatives[ForestStand]([
                        Event[ForestStand](treatment=do_nothing, static_parameters={"n": 1}, tags={"first_type"}),
                        Sequence[ForestStand]([
                            Event[ForestStand](treatment=do_nothing, static_parameters={"n": 2}, tags={"second_type"}),
                            Event[ForestStand](
                                treatment=do_nothing,
                                static_parameters={"n": 3},
                                dynamic_parameters={
                                    "m": lambda s: (s.site_type_category.value if s.site_type_category is not None else 0) + 100
                                },
                                tags={"third_type"},
                            ),
                            Harvest20percent(),
                            FirstThinningMineralSoils()
                        ]),
                    ])
                ])
        ],
        transition=Transition(grow_acta_fn, 5, db_output_state=False, db_output_cd=False),
        end_condition=ForestCondition(lambda x: x.unit.relative_time > 30),
    )
)
