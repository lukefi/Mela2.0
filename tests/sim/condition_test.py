from typing import Any
import unittest

from lukefi.metsi.domain.conditions import _get_tag_last_run, _get_treatment_last_run
from lukefi.metsi.sim.collected_data import CollectedData, OpTuple
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.generators import Alternatives, Sequence, Event
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.treatment import PreparedTreatment, Treatment
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
        def _step(x: ToyModel) -> OpTuple[ToyModel]:
            computational_unit = x
            computational_unit.value += 1
            return computational_unit, []

        step = Treatment(_step, "step")

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

    def test_treatment_last_run(self):

        def operation1_fn(x: Any) -> OpTuple:
            return x, []

        def operation2_fn(x: Any) -> OpTuple:
            return x, []

        def operation3_fn(x: Any) -> OpTuple:
            return x, []

        def operation4_fn(x: Any) -> OpTuple:
            return x, []

        operation1 = Treatment(operation1_fn, "operation1")
        operation2 = Treatment(operation2_fn, "operation2")
        operation3 = Treatment(operation3_fn, "operation3")
        operation4 = Treatment(operation4_fn, "operation4")

        operation_history = [
            (1, "operation1", {}, set()),
            (2, "operation2", {}, set()),
            (3, "operation1", {}, set()),
            (4, "operation3", {}, set()),
            (5, "operation1", {}, set()),
            (6, "operation2", {}, set()),
            (7, "operation1", {}, set()),
            (8, "operation3", {}, set()),
            (9, "operation1", {}, set())
        ]

        self.assertEqual(_get_treatment_last_run(operation_history, operation1), 9)
        self.assertEqual(_get_treatment_last_run(operation_history, operation2), 6)
        self.assertEqual(_get_treatment_last_run(operation_history, operation3), 8)
        self.assertEqual(_get_treatment_last_run(operation_history, operation4), None)

    def test_tag_last_run(self):
        operation_history = [
            (1, "operation1", {}, {"1"}),
            (2, "operation2", {}, {"2"}),
            (3, "operation1", {}, {"1"}),
            (4, "operation3", {}, {"3"}),
            (5, "operation1", {}, {"1"}),
            (6, "operation2", {}, {"2"}),
            (7, "operation1", {}, {"1"}),
            (8, "operation3", {}, {"3"}),
            (9, "operation1", {}, {"1"})
        ]

        self.assertEqual(_get_tag_last_run(operation_history, "1"), 9)
        self.assertEqual(_get_tag_last_run(operation_history, "2"), 6)
        self.assertEqual(_get_tag_last_run(operation_history, "3"), 8)
        self.assertEqual(_get_tag_last_run(operation_history, "4"), None)

    def test_prepared_treatment_tag_combinations(self):
        def dummy_fn(x: ToyModel, **kwargs) -> tuple[ToyModel, list[CollectedData]]:
            _ = kwargs
            return x, []

        dummy_1 = Treatment(dummy_fn, "dummy", {"default_tag_1"})
        dummy_2 = Treatment(dummy_fn, "dummy", {"default_tag_2"})

        prepared_dummy_1 = PreparedTreatment(dummy_1, {"additional_tag_1", "additional_tag_2"})
        prepared_dummy_2 = PreparedTreatment(dummy_1, {"additional_tag_1", "additional_tag_3"})
        prepared_dummy_3 = PreparedTreatment(dummy_2, {"additional_tag_3", "additional_tag_4"})

        self.assertSetEqual({"default_tag_1", "additional_tag_1", "additional_tag_2"}, prepared_dummy_1.tags)
        self.assertSetEqual({"default_tag_1", "additional_tag_1", "additional_tag_3"}, prepared_dummy_2.tags)
        self.assertSetEqual({"default_tag_2", "additional_tag_3", "additional_tag_4"}, prepared_dummy_3.tags)
