from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.conditions import TimePoints
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.grow_motti_dll import grow_motti_dll_fn
from lukefi.metsi.domain.pre_ops import generate_reference_trees
from lukefi.metsi.app.metsi_control import AppConfiguration, MetsiControl
from lukefi.metsi.sim.sim_control import Preprocessing, Simulation
from lukefi.metsi.sim.transition import Transition
from lukefi.metsi.sim.instructions import SimulationInstruction
from examples.declarations.sqlite import sqlite_decl

from user_events import Mounding

control_structure = MetsiControl[ForestStand](
    app_configuration=AppConfiguration(
        state_format=StateFormat.XML,
        run_modes=[RunMode.PREPROCESS, RunMode.SIMULATE],
        sqlite_decl=sqlite_decl,
    ),
    preprocessing=Preprocessing[ForestStand](
        operations=[
            generate_reference_trees,
        ],
        params={
            generate_reference_trees: [
                {
                    "n_trees": 10,
                    "method": "weibull",
                    "debug": False
                }
            ]
        }),
    simulation=Simulation[ForestStand](
        instructions=[
            SimulationInstruction(
                conditions=[
                    TimePoints([2020])
                ],
                events=[Mounding()]
            ),
        ],
        transition=Transition(grow_motti_dll_fn, db_output_state=False, db_output_cd=False),
        end_condition=ForestCondition(lambda payload: payload.computational_unit.time > 2050))
)


__all__ = ['control_structure']
