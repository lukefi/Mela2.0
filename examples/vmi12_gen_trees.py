""" This declaration is used to define the output content of the preprocessing results """
from examples.declarations.export_prepro import default_csv
from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.pre_ops import generate_reference_trees
from lukefi.metsi.sim.sim_control import AppConfiguration, MetsiControl, Preprocessing


control_structure = MetsiControl[ForestStand](
    app_configuration=AppConfiguration(
        state_format=StateFormat.VMI12,
        run_modes=[RunMode.PREPROCESS, RunMode.EXPORT_PREPRO]
    ),
    preprocessing=Preprocessing(
        params={
            generate_reference_trees: [
                {
                    "n_trees": 10,
                    "method": "weibull"
                }
            ]
        },
        operations=[generate_reference_trees]
    ),
    export_prepro=default_csv
)

__all__ = ['control_structure']
