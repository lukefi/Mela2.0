from typing import Callable
import unittest
from lukefi.metsi.domain.conditions import MinimumTimeInterval
from lukefi.metsi.sim.collected_data import CollectedData
from lukefi.metsi.sim.operations import simple_processable_chain
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.generators import Alternatives, Sequence, Event
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.runners import evaluate_sequence as run_sequence, evaluate_sequence
from lukefi.metsi.sim.sim_configuration import SimConfiguration
from tests.test_utils import inc, parametrized_operation


class TestGenerators(unittest.TestCase):
    def test_yaml_declaration(self):
        declaration = {
            "simulation_instructions": [
                SimulationInstruction(
                    time_points=[0, 1],
                    events=Sequence([
                        Event(
                            treatment=inc
                        ),
                        Event(
                            treatment=inc
                        ),
                    ])
                )
            ]
        }
        config = SimConfiguration(**declaration)
        generator = config.full_tree_generators()
        result = generator.compose_nested()
        payload = SimulationPayload(
            computational_unit=0,
            operation_history=[]
        )
        computation_result = result.evaluate(payload)
        self.assertEqual(4, computation_result[0].computational_unit)

    def test_operation_run_constraints_success(self):
        declaration = {
            "simulation_instructions": [
                SimulationInstruction(
                    time_points=[1, 3],
                    events=Sequence([
                        Event(
                            preconditions=[
                                MinimumTimeInterval(2, inc)
                            ],
                            treatment=inc
                        )
                    ])
                )
            ]
        }
        config = SimConfiguration(**declaration)
        generator = config.full_tree_generators()
        result = generator.compose_nested()
        payload = SimulationPayload(
            computational_unit=0,
            operation_history=[]
        )
        computation_result = result.evaluate(payload)
        self.assertEqual(2, computation_result[0].computational_unit)

    def test_operation_run_constraints_fail(self):
        declaration = {
            "simulation_instructions": [
                SimulationInstruction(
                    time_points=[1, 3],
                    events=Sequence([
                        Event(
                            preconditions=[
                                MinimumTimeInterval(2, inc)
                            ],
                            treatment=inc
                        ),
                        Event(
                            preconditions=[
                                MinimumTimeInterval(2, inc)
                            ],
                            treatment=inc
                        )
                    ])
                )
            ]
        }
        config = SimConfiguration(**declaration)
        generator = config.full_tree_generators()
        result = generator.compose_nested()
        payload = SimulationPayload(computational_unit=0,
                                    operation_history=[])
        self.assertRaises(Exception, result.evaluate, payload)

    def test_nested_tree_generators(self):
        """Create a nested generators event tree. Use simple incrementation operation with starting value 0. Sequences
        and alternatives result in 4 branches with separately incremented values."""
        declaration = {
            "simulation_instructions": [
                SimulationInstruction(
                    time_points=[0],
                    events=Sequence([
                        Event(inc),
                        Sequence([
                            Event(inc)
                        ]),
                        Alternatives([
                            Event(inc),
                            Sequence([
                                Event(inc),
                                Alternatives([
                                    Event(inc),
                                    Sequence([
                                        Event(inc),
                                        Event(inc)
                                    ])
                                ])
                            ]),
                            Sequence([
                                Event(inc),
                                Event(inc),
                                Event(inc),
                                Event(inc)
                            ])
                        ]),
                        Event(inc),
                        Event(inc)
                    ])
                )
            ]
        }
        config = SimConfiguration(**declaration)
        generator = config.full_tree_generators()
        tree = generator.compose_nested()

        results = tree.evaluate(SimulationPayload(computational_unit=0, operation_history=[]))

        self.assertListEqual([5, 6, 7, 8], list(map(lambda result: result.computational_unit, results)))

    def test_nested_tree_generators_multiparameter_alternative(self):
        def increment(x, **y):
            return inc(x, **y)

        def inc_param(x, **y):
            return inc(x, **y)

        declaration = {
            "operation_params": {
                inc_param: [
                    {"incrementation": 2},
                    {"incrementation": 3}
                ]
            },
            "simulation_instructions": [
                SimulationInstruction(
                    time_points=[0],
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
            ]
        }
        config = SimConfiguration(**declaration)
        generator = config.full_tree_generators()
        tree = generator.compose_nested()

        results = tree.evaluate(SimulationPayload(computational_unit=0, operation_history=[]))

        self.assertListEqual([3, 4, 5], list(map(lambda result: result.computational_unit, results)))

    def test_alternatives_embedding_equivalence(self):
        """
        This test shows that alternatives with multiple single operations nested in alternatives is equivalent to
        sequences with single operations nested in alternatives.
        """
        declaration_one = {
            "simulation_instructions": [
                SimulationInstruction(
                    time_points=[0],
                    events=Sequence([
                        Event(inc),
                        Alternatives([
                            Alternatives([
                                Event(inc),
                                Event(inc)
                            ]),
                            Sequence([
                                Event(inc),
                                Event(inc)
                            ]),
                            Alternatives([
                                Event(inc),
                                Event(inc)
                            ])
                        ]),
                        Event(inc)
                    ])
                )
            ]
        }
        declaration_two = {
            "simulation_instructions": [
                SimulationInstruction(
                    time_points=[0],
                    events=Sequence([
                        Event(inc),
                        Alternatives([
                            Sequence([Event(inc)]),
                            Sequence([Event(inc)]),
                            Sequence([Event(inc), Event(inc)]),
                            Sequence([Event(inc)]),
                            Sequence([Event(inc)])
                        ]),
                        Event(inc)
                    ])
                )
            ]
        }
        configs = [
            SimConfiguration(**declaration_one),
            SimConfiguration(**declaration_two)
        ]
        generators = [config.full_tree_generators() for config in configs]
        trees = [generator.compose_nested() for generator in generators]

        results = (trees[0].evaluate(SimulationPayload(computational_unit=0, operation_history=[])),
                   trees[1].evaluate(SimulationPayload(computational_unit=0, operation_history=[])))

        self.assertListEqual(results[0], results[1])

    def test_simple_processable_chain_multiparameter_exception(self):
        operation_tags = ['param_oper']
        operation_params = {'param_oper': [{'amplify': True}, {'kissa123': 123}]}
        operation_lookup = {'param_oper': parametrized_operation}
        self.assertRaises(Exception,
                          simple_processable_chain,
                          operation_tags,
                          operation_params,
                          operation_lookup)

    def test_generate_time_series(self):
        declaration = {
            "simulation_instructions": [
                SimulationInstruction(
                    time_points=[0, 1, 4, 100, 1000, 8, 9],
                    events=Sequence([
                        Event(inc),
                        Event(inc)
                    ])
                ),
                SimulationInstruction(
                    time_points=[9, 8],
                    events=Sequence([
                        Event(inc),
                        Event(inc)
                    ])
                ),
                SimulationInstruction(
                    time_points=[4, 6, 10, 12],
                    events=Sequence([
                        Event(inc),
                        Event(inc)
                    ])
                )
            ]
        }
        result = SimConfiguration(**declaration)
        self.assertEqual([0, 1, 4, 6, 8, 9, 10, 12, 100, 1000], result.time_points)
