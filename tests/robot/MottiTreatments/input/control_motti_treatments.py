from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.domain.pre_ops import (
    compute_location_metadata,
    filter_stands,
    filter_trees,
    generate_reference_trees,
    scale_area_weight)
from lukefi.metsi.domain.natural_processes.grow_motti_dll import grow_motti_dll_fn
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.generators import Alternatives, Event, Sequence
from lukefi.metsi.sim.sim_configuration import Initialization, Transition
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.domain.natural_processes.motti_bootstrap import initialize_motti
from lukefi.metsi.sim.treatment import do_nothing
from user_events import (
    Harvest20percent,
    FirstThinningMineralSoils,
    PlantingPines,
    SaplingTreatmentMotti,
    EarlyCareMotti,
    FillinPlantingMotti,
    SeedlingDelayMotti,
    SeedtreeCutting,
)

Motti4DLL.load()

control_structure = {
    "app_configuration": {
        "state_format": "vmi13",
        "run_modes": ["preprocess", "simulate"],
    },
    "preprocessing_operations": [
        scale_area_weight,
        generate_reference_trees,
        compute_location_metadata,
        filter_stands,
        filter_trees,
    ],
    "preprocessing_params": {
        generate_reference_trees: [
            {
                "n_trees": 10,
                "method": "weibull",
                "debug": False,
                "delete_strata": False
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
                        FirstThinningMineralSoils(),


                    ]),
                    Sequence[ForestStand]([
                        PlantingPines(),
                        SeedlingDelayMotti(parameters={"istep": 1}),
                        EarlyCareMotti(parameters={"imode": 0}),

                    ]),
                    EarlyCareMotti(parameters={"imode": 0}),
                    Harvest20percent(),
                    SaplingTreatmentMotti(parameters={"remaining_n": {1: 1800,
                                                                      2: 1800,
                                                                      3: 1800,
                                                                      4: 1800,
                                                                      5: 1800,
                                                                      6: 1800, }}),
                    FillinPlantingMotti(parameters={"species": 1, "stems_per_ha": 400.0}),
                    SeedtreeCutting(parameters={
                        "cutting_method": 7,
                    })
                ])
            ]
        )
    ],
    "initialization": Initialization(initialize_motti),
    "transition": Transition(
        grow_motti_dll_fn,
        max_step=5,
        db_output=False,
    ),
    "end_condition": Condition[ForestStand](lambda x: x.computational_unit.year > 2030)
}

__all__ = ['control_structure']
