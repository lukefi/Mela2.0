import unittest

from lukefi.metsi.domain.conditions import MinimumTimeInterval
from lukefi.metsi.sim.collected_data import CollectedData
from lukefi.metsi.sim.operations import prepared_treatment
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.generators import Sequence, Event
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.event_tree import EventTree
from lukefi.metsi.sim.sim_configuration import SimConfiguration


def prep_inc(x: SimulationPayload[int]) -> tuple[SimulationPayload[int], list[CollectedData]]:
    x.computational_unit += 1
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
        results = self.root.evaluate(SimulationPayload(computational_unit=0, operation_history={}))
        self.assertListEqual([3, 3, 3, 3], [result.computational_unit for result in results])

    def test_sim_configuration(self):
        def fn1(x):
            return x

        def fn2(y):
            return y

        config = {
            'simulation_instructions': [
                SimulationInstruction(
                    time_points=[1, 2, 3],
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
                    time_points=[3, 4, 5],
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
                )
            ]
        }
        result = SimConfiguration(**config)
        self.assertListEqual([1, 2, 3, 4, 5], result.time_points)
        self.assertEqual(2, len(result.instructions))
