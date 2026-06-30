from lukefi.metsi.app.metsi_enum import RunMode, StateFormat, StrataOrigin
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.conditions import TimePoints
from lukefi.metsi.domain.events import DoNothing
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from lukefi.metsi.domain.pre_ops import filter_stands, filter_trees, generate_reference_trees
from lukefi.metsi.sim.generators import Sequence
from lukefi.metsi.sim.instructions import SimulationInstruction
from lukefi.metsi.sim.sim_control import AppConfiguration, MetsiControl, Preprocessing, Simulation
from examples.declarations.sqlite import sqlite_decl
from lukefi.metsi.sim.transition import Transition

control_structure = MetsiControl[ForestStand](
    app_configuration=AppConfiguration(
        state_format=StateFormat.VMI13,
        strata_origin=StrataOrigin.COMPUTED,
        run_modes=[RunMode.PREPROCESS, RunMode.EXPORT_PREPRO, RunMode.SIMULATE],
        sqlite_decl=sqlite_decl
    ),
    preprocessing=Preprocessing[ForestStand](
        operations=[
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
                    "remove": lambda stand: (stand.site_type_category is None) or (stand.site_type_category == 0)
                }
            ],
            filter_trees: [
                {
                    "predicate": lambda stand: ~(stand.reference_trees.sapling | (stand.reference_trees.stems_per_ha == 0))
                }
            ]
        }
    ),
    simulation=Simulation[ForestStand](
        instructions=[
            SimulationInstruction[ForestStand](
                conditions=[
                    TimePoints([2020])
                ],
                events=Sequence([
                    DoNothing()
                ])
            )
        ],
        transition=Transition(grow_acta_fn, db_output=False),
        end_condition=ForestCondition(lambda payload: payload.computational_unit.time > 2020)
    ),
    export_prepro={
        "csv": {}
    }
)
