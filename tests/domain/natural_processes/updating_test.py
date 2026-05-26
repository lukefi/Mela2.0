from lukefi.metsi.sim.operations import do_nothing
from lukefi.metsi.sim.treatment import PredeterminedTreatment
from lukefi.metsi.domain.natural_processes.updating import _get_next_step_and_treatments
from lukefi.metsi.data.model import ForestStand
import unittest


class TestUpdating(unittest.TestCase):
    def test_step_size_no_treatments(self):
        stand = ForestStand(time=2020)
        step, treatments = _get_next_step_and_treatments(stand, 2025)
        self.assertEqual(5, step)
        self.assertListEqual([], treatments)

    def test_step_size_with_treatments(self):
        stand = ForestStand(time=2020, predetermined_events=[PredeterminedTreatment(2022, "do_something", do_nothing),
                            PredeterminedTreatment(2022, "do_another_thing", do_nothing),
                            PredeterminedTreatment(2023, "do_final_thing", do_nothing)])

        step, treatments = _get_next_step_and_treatments(stand, 2025)
        self.assertEqual(2, step)
        self.assertListEqual(["do_something", "do_another_thing"], [treatment.name for treatment in treatments])

        stand.time = 2022

        step, treatments = _get_next_step_and_treatments(stand, 2025)
        self.assertEqual(1, step)
        self.assertListEqual(["do_final_thing"], [treatment.name for treatment in treatments])

        stand.time = 2023

        step, treatments = _get_next_step_and_treatments(stand, 2025)
        self.assertEqual(2, step)
        self.assertListEqual([], [treatment.name for treatment in treatments])
