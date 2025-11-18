from pathlib import Path
import unittest

from lukefi.metsi.app.file_io import read_control_module
from lukefi.metsi.domain.conditions import TimeSinceTreatment, RelativeTimePoints, TimePoints
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.generators import Alternatives, Event, Sequence
from lukefi.metsi.sim.sim_configuration import SimConfiguration
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.simulator import _simulate_unit
from lukefi.metsi.sim.treatment import Treatment
from tests.test_utils import collect_results
from tests.toy_model import ToyModel, ToyTransition, toy_inc, toy_inc_fn


class SimulatorTest(unittest.TestCase):
    def test_simulation_instructions_declaration(self):
        declaration = {
            "simulation_instructions": [
                SimulationInstruction(
                    events=Sequence([
                        Event(
                            treatment=toy_inc
                        ),
                        Event(
                            treatment=toy_inc
                        ),
                    ])
                )
            ],
            "end_condition": Condition[ToyModel](lambda x: x.computational_unit.time > 1),
            "transition": ToyTransition()
        }
        config = SimConfiguration[ToyModel](**declaration)
        payload = SimulationPayload(
            computational_unit=ToyModel("", 0),
            operation_history=[]
        )
        result = _simulate_unit(payload, config)
        self.assertEqual(1, len(result))
        self.assertEqual(4, result[0].computational_unit.value)

    def test_operation_run_constraints_success(self):
        declaration = {
            "simulation_instructions": [
                SimulationInstruction(
                    events=Sequence([
                        Event(
                            preconditions=[
                                TimeSinceTreatment(2, toy_inc)
                            ],
                            treatment=toy_inc
                        )
                    ]),
                    conditions=[
                        TimePoints([1, 3])
                    ]
                )
            ],
            "transition": ToyTransition(),
            "end_condition": Condition[ToyModel](lambda x: x.computational_unit.time > 3)
        }
        config = SimConfiguration(**declaration)
        payload = SimulationPayload(
            computational_unit=ToyModel("", 0),
            operation_history=[]
        )
        computation_result = _simulate_unit(payload, config)
        self.assertEqual(2, computation_result[0].computational_unit.value)

    def test_operation_run_constraints_fail(self):
        declaration = {
            "simulation_instructions": [
                SimulationInstruction(
                    conditions=[TimePoints([1, 3])],
                    events=Sequence([
                        Event(
                            preconditions=[
                                TimeSinceTreatment(2, toy_inc)
                            ],
                            treatment=toy_inc
                        ),
                        Event(
                            preconditions=[
                                TimeSinceTreatment(2, toy_inc)
                            ],
                            treatment=toy_inc
                        )
                    ])
                )
            ],
            "end_condition": Condition[ToyModel](lambda x: x.computational_unit.time > 3),
            "transition": ToyTransition()
        }
        config = SimConfiguration(**declaration)
        payload = SimulationPayload(computational_unit=ToyModel("", 0),
                                    operation_history=[])
        computation_result = _simulate_unit(payload, config)

        self.assertEqual(0, len(computation_result))

    def test_relative_time_points(self):
        declaration = {
            "simulation_instructions": [
                SimulationInstruction(
                    events=Sequence([
                        Event(
                            preconditions=[
                                TimeSinceTreatment(2, toy_inc)
                            ],
                            treatment=toy_inc
                        )
                    ]),
                    conditions=[
                        RelativeTimePoints([1, 3])
                    ]
                )
            ],
            "transition": ToyTransition(),
            "end_condition": Condition[ToyModel](lambda x: x.computational_unit.relative_time > 5)
        }
        config = SimConfiguration(**declaration)
        payload = SimulationPayload(
            computational_unit=ToyModel("", 0, time=200),
            operation_history=[]
        )
        computation_result = _simulate_unit(payload, config)
        self.assertEqual(2, computation_result[0].computational_unit.value)

    def test_nested_tree_generators(self):
        """Create a nested generators event tree. Use simple incrementation operation with starting value 0. Sequences
        and alternatives result in 4 branches with separately incremented values."""
        declaration = {
            "simulation_instructions": [
                SimulationInstruction(
                    conditions=[TimePoints([0])],
                    events=Sequence([
                        Event(toy_inc),
                        Sequence([
                            Event(toy_inc)
                        ]),
                        Alternatives([
                            Event(toy_inc),
                            Sequence([
                                Event(toy_inc),
                                Alternatives([
                                    Event(toy_inc),
                                    Sequence([
                                        Event(toy_inc),
                                        Event(toy_inc)
                                    ])
                                ])
                            ]),
                            Sequence([
                                Event(toy_inc),
                                Event(toy_inc),
                                Event(toy_inc),
                                Event(toy_inc)
                            ])
                        ]),
                        Event(toy_inc),
                        Event(toy_inc)
                    ])
                )
            ],
            "transition": ToyTransition(),
            "end_condition": Condition[ToyModel](lambda x: x.computational_unit.time > 0)
        }
        config = SimConfiguration(**declaration)

        results = _simulate_unit(
            SimulationPayload(
                computational_unit=ToyModel(
                    "",
                    0),
                operation_history=[]),
            config)

        self.assertListEqual([5, 6, 7, 8], list(map(lambda result: result.computational_unit.value, results)))

    def test_nested_tree_generators_multiparameter_alternative(self):
        def _increment(x, **y):
            return toy_inc_fn(x, **y)

        def _inc_param(x, **y):
            return toy_inc_fn(x, **y)

        increment = Treatment(_increment, "increment")
        inc_param = Treatment(_inc_param, "inc_param")

        declaration = {
            "simulation_instructions": [
                SimulationInstruction(
                    conditions=[TimePoints([0])],
                    events=Sequence([
                        Event(increment),
                        Alternatives([
                            Sequence([
                                Event(increment)
                            ]),
                            Alternatives([
                                Event(
                                    inc_param,
                                    parameters={
                                        "incrementation": 2
                                    }
                                ),
                                Event(
                                    inc_param,
                                    parameters={
                                        "incrementation": 3
                                    }
                                )

                            ]),
                        ]),
                        Event(increment)
                    ])
                )
            ],
            "end_condition": Condition[ToyModel](lambda x: x.computational_unit.time > 0),
            "transition": ToyTransition()
        }
        config = SimConfiguration(**declaration)

        results = _simulate_unit(
            SimulationPayload(
                computational_unit=ToyModel("", 0),
                operation_history=[]),
            config)

        self.assertListEqual([3, 4, 5], list(map(lambda result: result.computational_unit.value, results)))

    def test_alternatives_embedding_equivalence(self):
        """
        This test shows that alternatives with multiple single operations nested in alternatives is equivalent to
        sequences with single operations nested in alternatives.
        """
        declaration_one = {
            "simulation_instructions": [
                SimulationInstruction(
                    conditions=[TimePoints([0])],
                    events=Sequence([
                        Event(toy_inc),
                        Alternatives([
                            Alternatives([
                                Event(toy_inc),
                                Event(toy_inc)
                            ]),
                            Sequence([
                                Event(toy_inc),
                                Event(toy_inc)
                            ]),
                            Alternatives([
                                Event(toy_inc),
                                Event(toy_inc)
                            ])
                        ]),
                        Event(toy_inc)
                    ])
                )
            ],
            "transition": ToyTransition(),
            "end_condition": Condition[ToyModel](lambda x: x.computational_unit.time > 0)
        }
        declaration_two = {
            "simulation_instructions": [
                SimulationInstruction(
                    conditions=[TimePoints([0])],
                    events=Sequence([
                        Event(toy_inc),
                        Alternatives([
                            Sequence([Event(toy_inc)]),
                            Sequence([Event(toy_inc)]),
                            Sequence([Event(toy_inc), Event(toy_inc)]),
                            Sequence([Event(toy_inc)]),
                            Sequence([Event(toy_inc)])
                        ]),
                        Event(toy_inc)
                    ])
                )
            ],
            "transition": ToyTransition(),
            "end_condition": Condition[ToyModel](lambda x: x.computational_unit.time > 0)
        }
        configs = [
            SimConfiguration(**declaration_one),
            SimConfiguration(**declaration_two)
        ]

        results = (
            list(
                map(
                    lambda x: x.computational_unit.value,
                    _simulate_unit(
                        SimulationPayload(
                            computational_unit=ToyModel(
                                "",
                                0),
                            operation_history=[]),
                        configs[0]))),
            list(
                map(
                    lambda x: x.computational_unit.value,
                    _simulate_unit(
                        SimulationPayload(
                            computational_unit=ToyModel(
                                "",
                                0),
                            operation_history=[]),
                        configs[1]))))

        self.assertListEqual(results[0], results[1])

    def test_full_formation_evaluation_strategies_by_comparison(self):
        control_path = str(Path("tests",
                                "resources",
                                "runners_test",
                                "branching.py").resolve())
        declaration = read_control_module(control_path)
        config = SimConfiguration(**declaration)
        depth_payload = SimulationPayload(
            computational_unit=ToyModel("", 1),
            operation_history=[]
        )
        results_depth = collect_results(
            _simulate_unit(depth_payload, config))
        self.assertEqual(8, len(results_depth))

    def test_no_parameters_propagation(self):
        control_path = str(Path("tests",
                                "resources",
                                "runners_test",
                                "no_parameters.py").resolve())
        declaration = read_control_module(control_path)
        config = SimConfiguration(**declaration)
        initial = SimulationPayload(
            computational_unit=ToyModel("", 1),
            operation_history=[]
        )
        results = collect_results(
            _simulate_unit(initial, config))
        self.assertEqual(5, results[0].value)

    def test_parameters_propagation(self):
        control_path = str(Path("tests",
                                "resources",
                                "runners_test",
                                "parameters.py").resolve())
        declaration = read_control_module(control_path)
        config = SimConfiguration(**declaration)
        initial = SimulationPayload(
            computational_unit=ToyModel("", 1),
            operation_history=[]
        )
        results = collect_results(
            _simulate_unit(initial, config))
        self.assertEqual(9, results[0].value)

    def test_parameters_branching(self):
        control_path = str(Path("tests",
                                "resources",
                                "runners_test",
                                "parameters_branching.py").resolve())
        declaration = read_control_module(control_path)
        config = SimConfiguration(**declaration)
        initial = SimulationPayload(
            computational_unit=ToyModel("", 1),
            operation_history=[]
        )
        results = list(map(lambda x: x.value, collect_results(_simulate_unit(initial, config))))
        # do_nothing, do_nothing = 1
        # do_nothing, inc#1      = 2
        # do_nothing, inc#2      = 3
        # inc#1, do_nothing      = 2
        # inc#1, inc#1           = 3
        # inc#1, inc#2           = 4
        # inc#2, do_nothing      = 3
        # inc#2, inc#1           = 4
        # inc#2, inc#2           = 5
        expected = [1, 2, 3, 2, 3, 4, 3, 4, 5]
        self.assertEqual(expected, results)

    def test_multiple_instructions(self):
        declaration = {
            "simulation_instructions": [
                SimulationInstruction(
                    events=[
                        Event(toy_inc, parameters={
                            "incrementation": 1
                        })
                    ]
                ),
                SimulationInstruction(
                    events=[
                        Alternatives([
                            Event(toy_inc, parameters={
                                "incrementation": 2
                            }),
                            Event(toy_inc, parameters={
                                "incrementation": 3
                            }),
                        ])
                    ]
                )
            ],
            "transition": ToyTransition(),
            "end_condition": Condition[ToyModel](lambda x: x.computational_unit.time >= 3)
        }

        config = SimConfiguration[ToyModel](**declaration)
        payload = SimulationPayload[ToyModel](computational_unit=ToyModel("test", 0))

        results = _simulate_unit(payload, config)

        self.assertEqual(27, len(results))
        self.assertListEqual(
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 2],
             [0, 0, 2, 0], [0, 0, 2, 1], [0, 0, 2, 2], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 0, 2],
             [0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 1, 2], [0, 1, 2, 0], [0, 1, 2, 1], [0, 1, 2, 2],
             [0, 2, 0, 0], [0, 2, 0, 1], [0, 2, 0, 2], [0, 2, 1, 0], [0, 2, 1, 1], [0, 2, 1, 2],
             [0, 2, 2, 0], [0, 2, 2, 1], [0, 2, 2, 2],],
            [result.node_id for result in results]
        )
        self.assertListEqual(
            [3, 4, 5, 4, 5, 6, 5, 6, 7, 4, 5, 6, 5, 6, 7, 6, 7, 8, 5, 6, 7, 6, 7, 8, 7, 8, 9],
            [result.computational_unit.value for result in results]
        )
