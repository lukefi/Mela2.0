from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from lukefi.metsi.sim.instructions import ResimulationInstructions
from lukefi.metsi.sim.operations import do_nothing
from lukefi.metsi.domain.collected_data import NaturalProcessInfo
from examples.declarations.sqlite import sqlite_decl


control_structure = {
    "app_configuration": {
        "state_format": "db",  # options: fdm, vmi12, vmi13, xml, gpkg
        "run_modes": ["resimulate"],
        "simulation_output_file": "resimulation_results",
        "sqlite_decl": sqlite_decl,
    },
    "resimulation": ResimulationInstructions(
        transition=grow_acta_fn,
        schedules_file="selected_schedules.csv",
        treatment_map={
            "do_nothing": do_nothing,
        },
        collected_data={
            NaturalProcessInfo
        },
        data_type=ForestStand
    )
}

__all__ = ['control_structure']
