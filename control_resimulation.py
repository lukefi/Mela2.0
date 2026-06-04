from lukefi.metsi.domain.collected_data import NaturalProcessInfo
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from lukefi.metsi.sim.transition import Transition
from examples.declarations.sqlite import sqlite_decl


control_structure = {
    "app_configuration": {
        "state_format": "db",  # options: fdm, vmi12, vmi13, xml, gpkg
        "run_modes": ["resimulate"],
        "preprocessing_output_file": "preprocessing_results",
        "simulation_output_file": "simulation_results",
        "sqlite_decl": sqlite_decl,
    },
    # tarvitaanko? kannassa pitää olla vähintään nodet
    "transition": Transition(grow_acta_fn, 50, {NaturalProcessInfo}),
    # schedulet?
    "selected_schedules_file": "path_to_file.csv",
}

__all__ = ['control_structure']
