import unittest

from lukefi.mela2.domain.conditions import MinimumTimeInterval
from lukefi.mela2.sim.collected_data import CollectedData
from lukefi.mela2.sim.condition import Condition
from lukefi.mela2.sim.simulation_instruction import SimulationInstruction
from lukefi.mela2.sim.generators import Sequence, Event
from lukefi.mela2.sim.simulation_payload import SimulationPayload
from lukefi.mela2.sim.event_tree import EventTree
from lukefi.mela2.sim.sim_configuration import SimConfiguration
from tests.toy_model import ToyModel, ToyTransition


def prep_inc(x: SimulationPayload[ToyModel]) -> tuple[SimulationPayload[ToyModel], list[CollectedData]]:
    x.computational_unit.value += 1
    return x, []


class ComputationModelTest(unittest.TestCase):

    root = EventTree(prep_inc)
    root.branches = [
        EventTree(prep_inc),
        EventTree(prep_inc)
    ]

    root.branches[0].branches = [
        EventTree(prep_inc),
        EventTree(prep_inc)
    ]

    root.branches[1].branches = [
        EventTree(prep_inc),
        EventTree(prep_inc)
    ]

    def test_evaluator(self):
        results = self.root.evaluate(SimulationPayload(computational_unit=ToyModel("test", 0), operation_history={}))
        self.assertListEqual([3, 3, 3, 3], [result.computational_unit.value for result in results])

    def test_sim_configuration(self):
        def fn1(x):
            return x

        def fn2(y):
            return y

        config = {
            'simulation_instructions': [
                SimulationInstruction(
                    # time_points=[1, 2, 3],
                    events=[
                        Event(
                            preconditions=[
                                MinimumTimeInterval(5, fn1)
                            ],
                            treatment=fn1,
                            parameters={
                                'param1': 1
                            }
                        ),
                        Event(
                            treatment=fn2
                        )
                    ]
                ),
                SimulationInstruction(
                    # time_points=[3, 4, 5],
                    events=[
                        Sequence([
                            Event(
                                preconditions=[
                                    MinimumTimeInterval(5, fn1)
                                ],
                                treatment=fn1,
                                parameters={
                                    'param1': 1
                                }
                            ),
                            Event(
                                treatment=fn2
                            )
                        ])
                    ]
                ),
            ],
            "transition": ToyTransition(),
            "end_condition": Condition[ToyModel](lambda x: x.time > 5)
        }
        result = SimConfiguration(**config)
        self.assertEqual(2, len(result.instructions))
