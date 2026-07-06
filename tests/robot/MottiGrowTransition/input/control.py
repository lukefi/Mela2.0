from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.app.metsi_control import AppConfiguration, MetsiControl
from lukefi.metsi.core.sim_control import Preprocessing, Simulation
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.collected_data import NaturalProcessInfo
from lukefi.metsi.domain.conditions import TimePoints
from lukefi.metsi.domain.events import DoNothing
from lukefi.metsi.core.condition import Condition
from lukefi.metsi.core.transition import Transition
from lukefi.metsi.core.instructions import SimulationInstruction
from lukefi.metsi.domain.natural_processes.grow_motti_dll import grow_motti_dll_fn
from lukefi.metsi.domain.pre_ops import (
    compute_location_metadata,
    filter_stands,
    filter_trees,
    generate_reference_trees,
    scale_area_weight)


control_structure = MetsiControl[ForestStand](
    app_configuration=AppConfiguration(
        state_format=StateFormat.XML,
        run_modes=[RunMode.PREPROCESS, RunMode.SIMULATE]
    ),
    preprocessing=Preprocessing[ForestStand](
        operations=[
            scale_area_weight,
            generate_reference_trees,
            compute_location_metadata,
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
        }
    ),
    simulation=Simulation[ForestStand](
        instructions=[
            SimulationInstruction[ForestStand](
                conditions=[TimePoints([2025, 2030, 2035])],
                events=[
                    DoNothing()
                ]
            )
        ],
        transition=Transition(grow_motti_dll_fn,
                              max_step=5,
                              collected_data={NaturalProcessInfo},
                              name="grow_motti",
                              db_output_state=True,
                              db_output_cd=True),
        end_condition=Condition[ForestStand](lambda x: x.unit.year > 2030)
    )
)
