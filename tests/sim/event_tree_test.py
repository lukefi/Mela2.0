import unittest
from lukefi.metsi.sim.collected_data import CollectedData
from lukefi.metsi.sim.event_tree import EventTree
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from tests.toy_model import ToyModel


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
        results = self.root.evaluate(SimulationPayload(computational_unit=ToyModel("test", 0), operation_history=[]))
        self.assertListEqual([3, 3, 3, 3], [result.computational_unit.value for result in results])
