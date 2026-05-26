from unittest.mock import ANY
from unittest.mock import PropertyMock
from unittest.mock import call
from lukefi.metsi.domain.natural_processes.updating import update_to_year_fn
from unittest.mock import Mock
from lukefi.metsi.sim.operations import do_nothing
from lukefi.metsi.sim.treatment import PredeterminedTreatment
from lukefi.metsi.domain.natural_processes.updating import _get_step_and_treatments
from lukefi.metsi.data.model import ForestStand
import unittest


class TestUpdating(unittest.TestCase):
    def test_step_size_no_treatments(self):
        stand = ForestStand(time=2020)
        step, treatments = _get_step_and_treatments(stand, 2025)
        self.assertEqual(5, step)
        self.assertListEqual([], treatments)

    def test_step_size_with_treatments(self):
        stand = ForestStand(
            time=2020,
            predetermined_treatments=[
                PredeterminedTreatment(2022, "do_something", do_nothing),
                PredeterminedTreatment(2022, "do_another_thing", do_nothing),
                PredeterminedTreatment(2023, "do_final_thing", do_nothing)
            ]
        )

        step, treatments = _get_step_and_treatments(stand, 2025)
        self.assertEqual(2, step)
        self.assertListEqual(["do_something", "do_another_thing"], [treatment.name for treatment in treatments])

        stand.time = 2022

        step, treatments = _get_step_and_treatments(stand, 2025)
        self.assertEqual(1, step)
        self.assertListEqual(["do_final_thing"], [treatment.name for treatment in treatments])

        stand.time = 2023

        step, treatments = _get_step_and_treatments(stand, 2025)
        self.assertEqual(2, step)
        self.assertListEqual([], [treatment.name for treatment in treatments])

    def test_updating(self):
        treatment1 = Mock(time=2022)
        treatment1.side_effect = lambda x: (x, [])
        treatment2 = Mock(time=2023)
        treatment2.side_effect = lambda x: (x, [])
        stand = Mock(year=2020, predetermined_treatments=[treatment1, treatment2])
        transition = Mock()

        def transition_side(stand, step):
            stand.year += step
            return stand, []
        transition.side_effect = transition_side
        update_to_year_fn(stand, transition=transition, target_year=2025)

        self.assertEqual(2025, stand.year)
        treatment1.assert_called()
        treatment2.assert_called()
        transition.assert_has_calls([call(ANY, 2), call(ANY, 1), call(ANY, 2)])
