from user_events import Harvest20percent, FirstThinningMineralSoils
from examples.declarations.export_prepro import mela_and_default_csv
from lukefi.metsi.sim.treatment import do_nothing
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.sim_configuration import Transition
from lukefi.metsi.sim.generators import Alternatives, Event, Sequence
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from lukefi.metsi.domain.pre_ops import generate_reference_trees, preproc_filter, scale_area_weight


control_structure = {
    "app_configuration": {
        "state_format": "vmi13",  # options: fdm, vmi12, vmi13, xml, gpkg
        # "state_input_container": "csv",  # Only relevant with fdm state_format. Options: pickle, json
        "run_modes": ["preprocess", "export_prepro", "simulate"]
    },
    "preprocessing_operations": [
        scale_area_weight,
        generate_reference_trees,  # reference trees from strata, replaces existing reference trees
        preproc_filter,
    ],
    "preprocessing_params": {
        generate_reference_trees: [
            {
                "n_trees": 10,
                "method": "weibull",
                "debug": False
            }
        ],
        preproc_filter: [
            {
                "remove trees": (lambda trees: (trees.sapling != 0) | (trees.stems_per_ha == 0)),
                "remove stands": (lambda stand: (stand.site_type_category is None) or (stand.site_type_category == 0))
            }
        ]
    },
    "simulation_instructions": [
        SimulationInstruction(
            events=[
                Alternatives[ForestStand]([
                    Event[ForestStand](treatment=do_nothing, static_parameters={"n": 1}, tags={"first_type"}),
                    Sequence[ForestStand]([
                        Event[ForestStand](treatment=do_nothing, static_parameters={"n": 2}, tags={"second_type"}),
                        Event[ForestStand](
                            treatment=do_nothing,
                            static_parameters={"n": 3},
                            dynamic_parameters={
                                "m": lambda s: (s.site_type_category.value if s.site_type_category is not None else 0) + 100
                            },
                            tags={"third_type"},
                        ),
                        Harvest20percent(),
                        FirstThinningMineralSoils()
                    ]),
                ])
            ]
        )
    ],
    "transition": Transition(grow_acta_fn),
    "end_condition": ForestCondition(lambda x: x.computational_unit.relative_time > 30),
    "post_processing": {
        "operation_params": {
            do_nothing: [
                {"param": "value"}
            ]
        },
        "post_processing": [
            do_nothing
        ]
    },
    'export_prepro': mela_and_default_csv
}

__all__ = ['control_structure']
