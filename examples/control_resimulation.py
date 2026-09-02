from examples.declarations.sqlite import sqlite_decl
from lukefi.metsi.app.metsi_control import AppConfiguration, MetsiControl
from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.core.operations import do_nothing
from lukefi.metsi.core.sim_control import Resimulation
from lukefi.metsi.core.transition import Transition, Initialization
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.collected_data import NaturalProcessInfo
from lukefi.metsi.domain.natural_processes.motti_initialization import initialize_motti
from lukefi.metsi.domain.natural_processes.grow_motti import grow_motti_fn
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL

Motti4DLL.load()

control_structure = MetsiControl[ForestStand](
    app_configuration=AppConfiguration(
        state_format=StateFormat.DB,  # options: fdm, vmi12, vmi13, xml, gpkg
        run_modes=[RunMode.RESIMULATE],
        simulation_output_file="resimulation_results",
        sqlite_decl=sqlite_decl,
    ),
    resimulation=Resimulation[ForestStand](
        transition=Transition(grow_motti_fn,
                              max_step=5,
                              collected_data={NaturalProcessInfo},
                              name="grow_motti",
                              db_output_state=False,
                              db_output_cd=True,
                              initialization=Initialization(initialize_motti)
        ),
        schedules_file="examples/selected_schedules.csv",
        treatment_map={
            "do_nothing": do_nothing,
        },
        collected_data=None,
        output_treatment_state=True,
        output_treatment_cd=True
    )
)

__all__ = ['control_structure']
