from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.pre_ops import filter_stands, filter_trees, generate_reference_trees, scale_area_weight
from lukefi.metsi.app.metsi_control import AppConfiguration, MetsiControl
from lukefi.metsi.sim.sim_control import Preprocessing


control_structure = MetsiControl[ForestStand](
    app_configuration=AppConfiguration(
        state_format=StateFormat.VMI13,
        run_modes=[RunMode.PREPROCESS, RunMode.EXPORT_PREPRO]
    ),
    preprocessing=Preprocessing[ForestStand](
        operations=[
            scale_area_weight,
            generate_reference_trees,
            filter_stands,
            filter_trees
        ],
        params={
            generate_reference_trees: [
                {
                    "n_trees": 10,
                    "method": "weibull",
                    "debug": False,
                    "delete_strata": True
                }
            ],
            filter_stands: [
                {
                    "remove": (lambda stand: (stand.site_type_category is None) or (stand.site_type_category == 0))
                }
            ],
            filter_trees: [
                {
                    "predicate": (lambda stand: ~((stand.reference_trees.sapling != 0) |
                                                  (stand.reference_trees.stems_per_ha == 0)))
                }
            ]
        },
    ),
    export_prepro={
        'csv': {},
        'csv_exp': {},
        'rst': {}
    }
)

__all__ = ['control_structure']
