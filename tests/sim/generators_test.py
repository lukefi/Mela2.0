import unittest
from lukefi.metsi.domain.conditions import MinimumTimeInterval, TimePoints
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.operations import simple_processable_chain
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.generators import Alternatives, Sequence, Event
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.sim_configuration import SimConfiguration
from lukefi.metsi.sim.simulator import _simulate_unit
from tests.toy_model import ToyModel, ToyTransition, parametrized_treatment, toy_inc


class TestGenerators(unittest.TestCase):
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
            "end_condition": Condition[ToyModel](lambda x: x.time > 1),
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
                                MinimumTimeInterval(2, toy_inc)
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
            "end_condition": Condition[ToyModel](lambda x: x.time > 3)
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
                                MinimumTimeInterval(2, toy_inc)
                            ],
                            treatment=toy_inc
                        ),
                        Event(
                            preconditions=[
                                MinimumTimeInterval(2, toy_inc)
                            ],
                            treatment=toy_inc
                        )
                    ])
                )
            ],
            "end_condition": Condition[ToyModel](lambda x: x.time > 3),
            "transition": ToyTransition()
        }
        config = SimConfiguration(**declaration)
        payload = SimulationPayload(computational_unit=ToyModel("", 0),
                                    operation_history=[])
        computation_result = _simulate_unit(payload, config)

        self.assertEqual(0, len(computation_result))

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
            "end_condition": Condition[ToyModel](lambda x: x.time > 0)
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
        def increment(x, **y):
            return toy_inc(x, **y)

        def inc_param(x, **y):
            return toy_inc(x, **y)

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
            "end_condition": Condition[ToyModel](lambda x: x.time > 0),
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
            "end_condition": Condition[ToyModel](lambda x: x.time > 0)
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
            "end_condition": Condition[ToyModel](lambda x: x.time > 0)
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

    def test_simple_processable_chain_multiparameter_exception(self):
        operation_tags = [parametrized_treatment]
        operation_params = {parametrized_treatment: [{'amplify': True}, {'kissa123': 123}]}
        self.assertRaises(Exception,
                          simple_processable_chain,
                          operation_tags,
                          operation_params)
