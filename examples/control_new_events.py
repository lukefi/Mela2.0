from lukefi.metsi.domain.conditions import TimePoints
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.grow_motti_dll import grow_motti_dll_fn
from lukefi.metsi.domain.natural_processes.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.domain.pre_ops import generate_reference_trees
from lukefi.metsi.sim.sim_configuration import Transition
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from examples.declarations.sqlite import sqlite_decl

from user_events import Mounding

Motti4DLL.load()

control_structure = {
    "app_configuration": {
        "state_format": "xml",
        "run_modes": ["preprocess", "simulate"],
        "sqlite_decl": sqlite_decl,
    },
    "preprocessing_operations": [
        generate_reference_trees,
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
            conditions=[
                TimePoints([2020])
            ],
            events=[Mounding()]
        ),
    ],
    "transition": Transition(grow_motti_dll_fn, db_output=False),
    "end_condition": ForestCondition(lambda payload: payload.computational_unit.time > 2050)
}


__all__ = ['control_structure']
