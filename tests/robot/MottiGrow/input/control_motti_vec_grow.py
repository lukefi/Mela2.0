from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.grow_motti_dll import grow_motti_dll_fn
from lukefi.metsi.domain.natural_processes.motti_bootstrap import initialize_motti
from lukefi.metsi.domain.pre_ops import compute_location_metadata, filter_stands, filter_trees, generate_reference_trees, scale_area_weight
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.generators import Alternatives, Event, Sequence
from lukefi.metsi.sim.instructions import SimulationInstruction
from lukefi.metsi.sim.operations import do_nothing
from lukefi.metsi.sim.sim_control import AppConfiguration, MetsiControl, Preprocessing, Simulation
from lukefi.metsi.sim.transition import Transition
from lukefi.metsi.sim.treatment import do_nothing


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
            SimulationInstruction(
                events=[
                    Alternatives([
                        Event(treatment=do_nothing, static_parameters={"n": 1}, tags={"first_type"}),
                        Sequence([
                            Event(treatment=do_nothing, static_parameters={"n": 2}, tags={"second_type"}),
                            Event(treatment=do_nothing, static_parameters={"n": 3}, tags={"third_type"}),
                        ])
                    ])
                ]
            )
        ],
        transition=Transition(
            grow_motti_dll_fn,
            max_step=5,
            db_output=False,
            init_fn=initialize_motti
        ),
        end_condition=Condition[ForestStand](lambda x: x.computational_unit.year > 2030)
    )
)
