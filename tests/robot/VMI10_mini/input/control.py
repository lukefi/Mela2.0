from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from lukefi.metsi.domain.pre_ops import filter_stands, filter_trees, scale_area_weight
from lukefi.metsi.app.metsi_control import AppConfiguration, MetsiControl
from lukefi.metsi.sim.generators import Alternatives, Event, Sequence
from lukefi.metsi.sim.sim_control import Preprocessing, Simulation
from lukefi.metsi.sim.transition import Transition
from lukefi.metsi.sim.instructions import SimulationInstruction
from lukefi.metsi.sim.treatment import do_nothing
from examples.declarations.export_prepro import mela_decl


control_structure = MetsiControl[ForestStand](
    app_configuration=AppConfiguration(
        state_format=StateFormat.VMI10,
        measured_trees=True,
        run_modes=[RunMode.PREPROCESS, RunMode.EXPORT_PREPRO, RunMode.SIMULATE]
    ),
    preprocessing=Preprocessing[ForestStand](
        operations=[
            scale_area_weight,
            filter_stands,
            filter_trees,
        ],
        params={
            filter_stands: [
                {
                    "remove": (lambda stand: (stand.site_type_category is None) or (stand.site_type_category == 0))
                }
            ],
            filter_trees: [
                {
                    "predicate": (lambda stand: ~((stand.reference_trees.sapling != 0) |
                                                  (stand.reference_trees.stems_per_ha == 0))),
                }
            ],
        }),
    simulation=Simulation[ForestStand](
        instructions=[
            SimulationInstruction(
                events=[
                    Alternatives([
                        Event(treatment=do_nothing, static_parameters={"n": 1}, tags={"first_type"}),
                        Sequence([
                            Event(treatment=do_nothing, static_parameters={"n": 2}, tags={"second_type"}),
                            Event(treatment=do_nothing, static_parameters={"n": 3}, tags={"third_type"})
                        ])
                    ])
                ]
            )
        ],
        transition=Transition(grow_acta_fn, db_output_state=False, db_output_cd=False),
        end_condition=ForestCondition(lambda x: x.unit.year >= 2050)),
    export_prepro={
        'csv': {},
        'rst': mela_decl
    }
)

__all__ = ['control_structure']
