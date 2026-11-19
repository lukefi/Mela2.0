import numpy as np

from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.app.metsi_control import AppConfiguration, MetsiControl
from lukefi.metsi.core.utils.select_units import ProfileXMode, SelectionTarget, TargetType, SelectionSet
from lukefi.metsi.data.enums.internal import Storey, TreeManagementCategory
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.domain.pre_ops import (
    filter_stands,
    filter_trees,
    generate_reference_trees,
    scale_area_weight)
from lukefi.metsi.domain.natural_processes.grow_motti import grow_motti_fn
from lukefi.metsi.domain.natural_processes.motti_initialization import motti_init
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.core.condition import Condition
from lukefi.metsi.core.generators import Alternatives, Event
from lukefi.metsi.core.transition import Transition
from lukefi.metsi.core.instructions import SimulationInstruction
from lukefi.metsi.core.sim_control import Preprocessing, Simulation
from lukefi.metsi.domain.forestry_treatments.mark_trees import mark_trees


Motti4DLL.load()

tree_selection = {
    # Selecting over 50% of stems enables markings of sapling and updating of motti ut vector.
    "target": SelectionTarget(TargetType.RELATIVE, "stems_per_ha", 0.6),
    "sets": [
        SelectionSet[ForestStand, ReferenceTrees](
            lambda _, t: np.ones(t.size, dtype=bool),
            order_var="breast_height_diameter",
            target_var="stems_per_ha",
            target_type=TargetType.RELATIVE,
            target_amount=1.0,
            profile_x=[0, 1],
            profile_y=[0.2, 0.2],
            profile_xmode=ProfileXMode.RELATIVE
        )
    ]
}

CUSTOM_TREE_TYPE_FOR_ROBOT_TESTING = 'C'
static_parameters={
    "tree_selection": tree_selection,
    "select_from_all": True,
    "mode": "odds_units",
    "attributes": {
        "tree_type": CUSTOM_TREE_TYPE_FOR_ROBOT_TESTING,
        "management_category": TreeManagementCategory.RETENTION_TREE,
        "storey": Storey.RETENTION
    }
}

# Customised user event for testing workflow of mark_trees treatment in Motti context.
mark_trees_event_with_motti = Event(
    treatment=mark_trees,
    static_parameters=static_parameters
)


control_structure = MetsiControl[ForestStand](
    app_configuration=AppConfiguration(
        state_format=StateFormat.VMI13,
        run_modes=[RunMode.PREPROCESS, RunMode.SIMULATE],
    ),
    preprocessing=Preprocessing[ForestStand](
        operations=[
            scale_area_weight(),
            generate_reference_trees(
                n_trees=10,
                method="weibull",
                delete_strata=False
            ),
            filter_stands(
                remove=(lambda stand: (stand.site_type_category is None) or (stand.site_type_category == 0))
            ),
            filter_trees(
                predicate=(lambda stand: ~((stand.reference_trees.sapling != 0) |
                                           (stand.reference_trees.stems_per_ha == 0)))
            )
        ],
    ),
    simulation=Simulation[ForestStand](
        instructions=[
            SimulationInstruction(
                events=[
                    Alternatives[ForestStand]([
                        mark_trees_event_with_motti
                    ])
                ]
            )
        ],
        transition=Transition(
            grow_motti_fn,
            max_step=5,
            db_output_state=False,
            db_output_cd=False,
        ),
        initialization=motti_init,
        end_condition=Condition[ForestStand](lambda x: x.unit.year > 2028)
    )
)

__all__ = ['control_structure']
