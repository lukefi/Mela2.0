from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.app.metsi_control import AppConfiguration, MetsiControl
from lukefi.metsi.core.sim_control import Preprocessing, Simulation
from lukefi.metsi.core.treatment import Treatment
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.collected_data import NaturalProcessInfo
from lukefi.metsi.domain.conditions import TimePoints
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from lukefi.metsi.domain.natural_processes.grow_metsi import grow_metsi_fn
from lukefi.metsi.domain.pre_ops import filter_stands, filter_trees, generate_reference_trees, scale_area_weight
from lukefi.metsi.core.condition import Condition
from lukefi.metsi.core.generators import Alternatives, Event
from lukefi.metsi.core.operations import do_nothing
from lukefi.metsi.core.transition import Transition
from lukefi.metsi.core.instructions import SimulationInstruction


control_structure = MetsiControl[ForestStand](
    app_configuration=AppConfiguration(
        state_format=StateFormat.VMI13,
        run_modes=[RunMode.PREPROCESS, RunMode.SIMULATE]
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
    simulation=Simulation[ForestStand](
        instructions=[
            SimulationInstruction(
                conditions=[TimePoints([2018, 2023, 2028])],
                events=[
                    Alternatives([
                        Event(Treatment(lambda x: grow_acta_fn(x, 5), "grow_acta", set(), {NaturalProcessInfo})),
                        Event(Treatment(lambda x: grow_metsi_fn(x, 5), "grow_metsi", set(), {NaturalProcessInfo}))
                    ])
                ]
            )
        ],
        transition=Transition(do_nothing, 5, db_output_state=False, db_output_cd=False),
        end_condition=Condition[ForestStand](lambda x: x.unit.time > 2028)
    )
)
