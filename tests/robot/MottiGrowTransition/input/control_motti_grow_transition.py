from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.app.metsi_control import AppConfiguration, MetsiControl
from lukefi.metsi.core.generators import Event
from lukefi.metsi.core.treatment import do_nothing
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.collected_data import NaturalProcessInfo
from lukefi.metsi.domain.conditions import TimePoints
from lukefi.metsi.domain.natural_processes.grow_motti import grow_motti_fn
from lukefi.metsi.domain.natural_processes.motti_initialization import motti_init
from lukefi.metsi.domain.pre_ops import (
    compute_location_metadata,
    filter_stands,
    filter_trees,
    generate_reference_trees,
    scale_area_weight,
    supplement_storey_information)
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.core.sim_control import Preprocessing, PreprocessingOperation, Simulation
from lukefi.metsi.core.condition import Condition
from lukefi.metsi.core.transition import Transition
from lukefi.metsi.core.instructions import SimulationInstruction

Motti4DLL.load()

control_structure = MetsiControl[ForestStand](
    app_configuration=AppConfiguration(
        state_format=StateFormat.XML,
        run_modes=[RunMode.PREPROCESS, RunMode.SIMULATE]
    ),
    preprocessing=Preprocessing[ForestStand](
        operations=[
            PreprocessingOperation(scale_area_weight),
            PreprocessingOperation(generate_reference_trees, {
                "n_trees": 10,
                "method": "weibull",
                "debug": False,
                "delete_strata": False
            }),
            PreprocessingOperation(compute_location_metadata),
            PreprocessingOperation(filter_stands, {
                "remove": (lambda stand: (stand.site_type_category is None) or (stand.site_type_category == 0))
            }),
            PreprocessingOperation(filter_trees, {
                "predicate": (lambda stand: ~((stand.reference_trees.sapling != 0) |
                                              (stand.reference_trees.stems_per_ha == 0)))
            }),
            PreprocessingOperation(supplement_storey_information)
        ],
    ),
    simulation=Simulation[ForestStand](
        instructions=[
            SimulationInstruction[ForestStand](
                conditions=[TimePoints([2025, 2030, 2035])],
                events=[
                    Event(do_nothing)
                ]
            )
        ],
        transition=Transition(grow_motti_fn,
                              max_step=5,
                              collected_data={NaturalProcessInfo},
                              name="grow_motti",
                              db_output_state=True,
                              db_output_cd=True
                              ),
        initialization=motti_init,
        end_condition=Condition[ForestStand](lambda x: x.unit.year > 2030)
    )
)
