import unittest

from lukefi.metsi.core.collected_data import OpTuple
from lukefi.metsi.core.exceptions import CancelTreatment
from lukefi.metsi.core.generators import Event
from lukefi.metsi.core.model import ComputationalUnit
from lukefi.metsi.core.simulation_payload import SimulationPayload
from lukefi.metsi.core.treatment import Treatment
from tests.toy_model import ToyModel


def dummy_treatment_fn[T:ComputationalUnit](unit: T, *, cancel: bool = False) -> OpTuple[T]:
    if cancel:
        raise CancelTreatment("Canceled by parameter")
    return unit, []


dummy_treatment = Treatment(dummy_treatment_fn, "dummy")


class TreatmentTest(unittest.TestCase):
    def test_cancel_treatment(self):
        successful_event = Event(dummy_treatment)
        cancel_event = Event(dummy_treatment, static_parameters={"cancel": True})

        unit = ToyModel("unit", 0, 0)
        payload = SimulationPayload(unit)

        i = 0
        for _ in successful_event.evaluate(payload):
            i += 1

        self.assertEqual(i, 1)

        i = 0
        for _ in cancel_event.evaluate(payload):
            i += 1

        self.assertEqual(i, 0)
