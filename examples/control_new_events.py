from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.app.metsi_control import AppConfiguration, MetsiControl
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.conditions import TimePoints
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.grow_motti import grow_motti_fn
from lukefi.metsi.domain.natural_processes.motti_initialization import initialize_motti
from lukefi.metsi.domain.pre_ops import generate_reference_trees
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.core.sim_control import Preprocessing, Simulation
from lukefi.metsi.core.transition import Transition, Initialization
from lukefi.metsi.core.instructions import SimulationInstruction

from examples.declarations.sqlite import sqlite_decl
from user_events import Mounding


Motti4DLL.load()


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
        transition=Transition(grow_motti_fn,
                              db_output_state=False,
                              db_output_cd=False,
                              initialization=Initialization(initialize_motti)),
        end_condition=ForestCondition(lambda payload: payload.unit.time > 2050))
)


__all__ = ['control_structure']
