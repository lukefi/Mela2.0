from unittest.mock import ANY
from unittest.mock import call
from unittest.mock import Mock
import unittest
from lukefi.metsi.domain.natural_processes.updating import update_to_year_fn
from lukefi.metsi.sim.operations import do_nothing
from lukefi.metsi.sim.treatment import PredeterminedTreatment
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.updating import get_step_and_treatments


class TestUpdating(unittest.TestCase):
    def test_step_size_no_treatments(self):
        stand = ForestStand(time=2020)
        step, treatments = get_step_and_treatments(stand, 2025)
        self.assertEqual(5, step)
        self.assertListEqual([], treatments)

    def test_step_size_with_treatments(self):
        stand = ForestStand(
            time=2020,
            predetermined_treatments=[
                (2022, PredeterminedTreatment("do_something", do_nothing)),
                (2022, PredeterminedTreatment("do_another_thing", do_nothing)),
                (2023, PredeterminedTreatment("do_final_thing", do_nothing))
            ]
        )

        step, treatments = get_step_and_treatments(stand, 2025)
        self.assertEqual(2, step)
        self.assertListEqual(["do_something", "do_another_thing"], [treatment.name for treatment in treatments])

        stand.time = 2022

        step, treatments = get_step_and_treatments(stand, 2025)
        self.assertEqual(1, step)
        self.assertListEqual(["do_final_thing"], [treatment.name for treatment in treatments])

        stand.time = 2023

        step, treatments = get_step_and_treatments(stand, 2025)
        self.assertEqual(2, step)
        self.assertListEqual([], [treatment.name for treatment in treatments])

    def test_updating(self):
        treatment1 = Mock()
        treatment1.side_effect = lambda x: (x, [])
        treatment2 = Mock()
        treatment2.side_effect = lambda x: (x, [])
        stand = Mock(time=2020, predetermined_treatments=[(2022, treatment1), (2023, treatment2)])
        transition = Mock()

        def transition_side(stand, step):
            stand.time += step
            return stand, []
        transition.side_effect = transition_side
        update_to_year_fn(stand, transition=transition, target_year=2025)

        self.assertEqual(2025, stand.time)
        treatment1.assert_called_once()
        treatment2.assert_called_once()
        transition.assert_has_calls([call(ANY, 2), call(ANY, 1), call(ANY, 2)])
