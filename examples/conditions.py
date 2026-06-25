from lukefi.metsi.data.enums.internal import CuttingMethod
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from lukefi.metsi.sim.generators import Alternatives, Sequence, Event
from lukefi.metsi.sim.transition import Transition
from lukefi.metsi.sim.instructions import SimulationInstruction
from lukefi.metsi.domain.conditions import TimePoints
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.forestry_types import ForestOpPayload
from lukefi.metsi.domain.pre_ops import filter_stands, filter_trees, generate_reference_trees, scale_area_weight
from lukefi.metsi.sim.treatment import Treatment


def do_a_thing(x):
    """A treatment of some kind."""
    return x, []


def do_another_thing(x):
    """Another type of treatment."""
    return x, []


def do_yet_another_thing(x):
    """Yeat another type of treatment."""
    return x, []


def first_condition_check(x: ForestOpPayload):
    # Some complex condition check here.
    _ = x
    return True


def second_condition_check(x: ForestOpPayload):
    # Some complex condition check here.
    _ = x
    return True


# Conditions can be created by wrapping a predicate function:
first_condition = ForestCondition(first_condition_check)
second_condition = ForestCondition(second_condition_check)


# Conditions can also be created with the decorator syntax:
@ForestCondition
def third_condition(x: ForestOpPayload):
    # Some complex condition check here.
    _ = x
    return True


control_structure = {
    "app_configuration": {
        "state_format": "vmi13",  # options: fdm, vmi12, vmi13, xml, gpkg
        "run_modes": ["preprocess", "simulate"]
    },
    "preprocessing_operations": [
        scale_area_weight,
        generate_reference_trees,  # reference trees from strata, replaces existing reference trees
        filter_stands,
        filter_trees
    ],
    "preprocessing_params": {
        generate_reference_trees: [
            {
                "n_trees": 10,
                "method": "weibull",
                "debug": False
            }
        ],
        filter_stands: [
            {
                "remove": lambda stand: (stand.site_type_category is None) or (stand.site_type_category == 0)
            }
        ],
        filter_trees: [
            {
                "predicate": lambda stand: ~(stand.reference_trees.sapling | (stand.reference_trees.stems_per_ha == 0))
            }
        ],
    },
    "simulation_instructions": [
        SimulationInstruction(
            conditions=[TimePoints([2025, 2030, 2035])],
            events=[
                Sequence([
                    Alternatives([
                        Event(Treatment(do_a_thing),
                              preconditions=[
                                  # Conditions can be combined with | and & operators.
                                  # Here do_a_thing will be performed if the year is 2025 or any time that the last
                                  # cutting method was 1.
                                  ForestCondition(lambda x: x.computational_unit.time == 2025) |
                                  ForestCondition(
                                      lambda x: x.computational_unit.method_of_last_cutting == CuttingMethod.THINNING)
                        ]),
                        Event(Treatment(do_another_thing),
                              preconditions=[
                                  # Combined conditions can also be expressed with just one lambda:.
                                  # This time do_another_thing will be performed the year 2030 for all non-auxiliary
                                  # stands.
                                  ForestCondition(
                                      lambda x: (x.computational_unit.time == 2030) and
                                      (not x.computational_unit.auxiliary_stand))
                        ]),
                        Event(Treatment(do_yet_another_thing),
                              # More complex conditions can be formulated in separate modules, such as pre-made
                              # libraries, and combined freely in non-trivial ways.
                              preconditions=[
                                  (first_condition & second_condition) | (third_condition)
                        ])
                    ])
                ])
            ]
        )
    ],
    "transition": Transition(grow_acta_fn, db_output=False),
    "end_condition": ForestCondition(lambda payload: payload.computational_unit.time > 2035)
}

__all__ = ['control_structure']
