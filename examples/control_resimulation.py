from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.grow_motti_dll import grow_motti_dll_fn
from lukefi.metsi.sim.instructions import ResimulationInstructions
from lukefi.metsi.sim.operations import do_nothing
from lukefi.metsi.domain.collected_data import NaturalProcessInfo
from examples.declarations.sqlite import sqlite_decl
from lukefi.metsi.sim.transition import Transition


control_structure = {
    "app_configuration": {
        "state_format": "db",  # options: fdm, vmi12, vmi13, xml, gpkg
        "run_modes": ["resimulate"],
        "simulation_output_file": "resimulation_results",
        "sqlite_decl": sqlite_decl,
    },
    "resimulation": ResimulationInstructions[ForestStand](
        transition=Transition(grow_motti_dll_fn,
                              max_step=5,
                              collected_data={NaturalProcessInfo},
                              name="grow_motti"),
        schedules_file="selected_schedules.csv",
        treatment_map={
            "do_nothing": do_nothing,
        }
    )
}

__all__ = ['control_structure']
