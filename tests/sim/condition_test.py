from typing import Any
import unittest

from lukefi.metsi.domain.conditions import _get_operation_last_run
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.generators import Alternatives, Sequence, Event
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from tests.toy_model import ToyModel


class ConditionTest(unittest.TestCase):
    def test_condition_combinations(self):

        @Condition[ToyModel]
        def c2(x: ToyModel) -> bool:
            return x.value < 5

        c1 = Condition[ToyModel](lambda x: x.time >= 2)

        c_and = c1 & c2
        c_or = c1 | c2

        self.assertTrue(c_and(ToyModel("", 4, 2)))
        self.assertFalse(c_and(ToyModel("", 4, 1)))
        self.assertFalse(c_and(ToyModel("", 5, 2)))
        self.assertFalse(c_and(ToyModel("", 6, 1)))

        self.assertTrue(c_or(ToyModel("", 4, 3)))
        self.assertTrue(c_or(ToyModel("", 3, 1)))
        self.assertTrue(c_or(ToyModel("", 6, 5)))
        self.assertFalse(c_or(ToyModel("", 6, 1)))

    def test_condition_checking(self):
        def step(x: ToyModel) -> OpTuple[ToyModel]:
            computational_unit = x
            computational_unit.value += 1
            return computational_unit, []

        generator = Alternatives[ToyModel]([
            Sequence([
                Event(step, preconditions=[Condition(lambda x: x.computational_unit.value <= 2)]),
                Event(step, preconditions=[Condition(lambda x: x.computational_unit.value >= 2)]),
                Event(step, postconditions=[Condition(lambda x: x.computational_unit.value == 4)]),
            ]),
            Sequence([
                Event(step, preconditions=[Condition(lambda x: x.computational_unit.value < 2)]),
                Event(step, preconditions=[Condition(lambda x: x.computational_unit.value >= 2)]),
                Event(step, postconditions=[Condition(lambda x: x.computational_unit.value == 3)]),
            ]),
            Sequence([
                Event(step, postconditions=[Condition(lambda x: x.computational_unit.value == 2)]),
                Event(step, postconditions=[Condition(lambda x: x.computational_unit.value < 5)]),
            ]),
            Event(step, preconditions=[Condition(lambda _: True)]),
            Event(step, preconditions=[Condition(lambda _: False)]),
            Event(step, postconditions=[Condition(lambda _: True)]),
            Event(step, postconditions=[Condition(lambda _: False)]),
        ])

        roots = generator.compose_nested()
        result: list[SimulationPayload[ToyModel]] = []
        for root in roots:
            result.extend(root.evaluate(SimulationPayload(
                computational_unit=ToyModel("", 1),
                operation_history=[])))

        self.assertEqual(len(list(result)), 4)
        self.assertEqual(result[0].computational_unit.value, 4)
        self.assertEqual(result[1].computational_unit.value, 3)
        self.assertEqual(result[2].computational_unit.value, 2)
        self.assertEqual(result[3].computational_unit.value, 2)

    def test_operation_last_run(self):

        def operation1(x: Any) -> OpTuple:
            return x, []

        def operation2(x: Any) -> OpTuple:
            return x, []

        def operation3(x: Any) -> OpTuple:
            return x, []

        def operation4(x: Any) -> OpTuple:
            return x, []

        operation_history = [
            (1, operation1, {}),
            (2, operation2, {}),
            (3, operation1, {}),
            (4, operation3, {}),
            (5, operation1, {}),
            (6, operation2, {}),
            (7, operation1, {}),
            (8, operation3, {}),
            (9, operation1, {})
        ]

        self.assertEqual(_get_operation_last_run(operation_history, operation1), 9)
        self.assertEqual(_get_operation_last_run(operation_history, operation2), 6)
        self.assertEqual(_get_operation_last_run(operation_history, operation3), 8)
        self.assertEqual(_get_operation_last_run(operation_history, operation4), None)
