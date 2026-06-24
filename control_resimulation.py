from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from lukefi.metsi.sim.operations import do_nothing
from lukefi.metsi.domain.natural_processes.grow_metsi import grow_metsi_fn
from examples.declarations.sqlite import sqlite_decl


control_structure = {
    "app_configuration": {
        "state_format": "db",  # options: fdm, vmi12, vmi13, xml, gpkg
        "run_modes": ["resimulate"],
        "simulation_output_file": "resimulation_results",
        "sqlite_decl": sqlite_decl,
    },
    # tarvitaanko? kannassa pitää olla vähintään nodet
    "transition": grow_acta_fn,
    # schedulet?
    "selected_schedules_file": "selected_schedules.csv",
    "treatment_map": {
        "do_nothing": do_nothing,
        "grow_acta": grow_acta_fn,
        "grow_metsi": grow_metsi_fn
    }
}

__all__ = ['control_structure']
