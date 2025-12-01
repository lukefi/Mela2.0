import unittest
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.runners import evaluate_sequence
from tests.test_utils import raises, identity, none
from tests.toy_model import ToyModel


class RunnersTest(unittest.TestCase):
    def test_sequence_success(self):
        payload = SimulationPayload(computational_unit=ToyModel("", 1))
        result = evaluate_sequence(
            payload,
            identity,
            none
        )
        self.assertEqual(None, result)

    def test_sequence_failure(self):
        payload = SimulationPayload(computational_unit=ToyModel("", 1))

        def prepared_function():
            return evaluate_sequence(payload, identity, raises, identity)

        self.assertRaises(Exception, prepared_function)
