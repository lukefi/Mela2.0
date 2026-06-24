from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
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
    "transition": grow_acta_fn,
    # schedulet?
    "selected_schedules_file": "selected_schedules.csv",
}

__all__ = ['control_structure']
