from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.core.sim_control import Preprocessing, PreprocessingOperation
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.pre_ops import filter_stands, filter_trees, generate_reference_trees, scale_area_weight
from lukefi.metsi.app.metsi_control import AppConfiguration, MetsiControl


control_structure = MetsiControl[ForestStand](
    app_configuration=AppConfiguration(
        state_format=StateFormat.VMI13,
        run_modes=[RunMode.PREPROCESS, RunMode.EXPORT_PREPRO]
    ),
    preprocessing=Preprocessing[ForestStand](
        operations=[
            PreprocessingOperation(scale_area_weight),
            PreprocessingOperation(generate_reference_trees, {
                "n_trees": 10,
                "method": "weibull",
                "debug": False,
                "delete_strata": True
            }),
            PreprocessingOperation(filter_stands, {
                "remove": (lambda stand: (stand.site_type_category is None) or (stand.site_type_category == 0))
            }),
            PreprocessingOperation(filter_trees, {
                "predicate": (lambda stand: ~((stand.reference_trees.sapling != 0) |
                                              (stand.reference_trees.stems_per_ha == 0)))
            })
        ],
    ),
    export_prepro={
        'csv': {},
        'csv_exp': {},
        'rst': {}
    }
)

__all__ = ['control_structure']
