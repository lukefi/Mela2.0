import unittest
from unittest.mock import ANY, Mock, call

from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.operations import do_nothing
from lukefi.metsi.sim.sim_control import Updating
from lukefi.metsi.sim.transition import Transition
from lukefi.metsi.sim.treatment import PredeterminedTreatment
from lukefi.metsi.sim.updating import get_step_and_treatments, update_units


class UpdatingTest(unittest.TestCase):
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
        self.assertListEqual([], [treatment.name for treatment in treatments])

        stand.time = 2022

        step, treatments = get_step_and_treatments(stand, 2025)
        self.assertEqual(1, step)
        self.assertListEqual(["do_something", "do_another_thing"], [treatment.name for treatment in treatments])

        stand.time = 2023

        step, treatments = get_step_and_treatments(stand, 2025)
        self.assertEqual(2, step)
        self.assertListEqual(["do_final_thing"], [treatment.name for treatment in treatments])

        stand.time = 2025

        step, treatments = get_step_and_treatments(stand, 2025)
        self.assertEqual(0, step)
        self.assertListEqual([], [treatment.name for treatment in treatments])

    def test_updating_run_mode(self):
        treatment1 = Mock()
        treatment1.side_effect = lambda x: (x, [])
        treatment2 = Mock()
        treatment2.side_effect = lambda x: (x, [])
        stand1 = Mock(time=2020, predetermined_treatments=[(2022, treatment1), (2023, treatment2)])
        stand2 = Mock(time=2018, predetermined_treatments=[(2022, treatment1), (2023, treatment2)])
        stands = [stand1, stand2]
        transition_fn = Mock()

        def transition_side(stand, step):
            stand.time += step
            return stand, []

        transition_fn.side_effect = transition_side
        transition = Transition(transition_fn, name="transition")
        instructions = Updating(target_time=2025,
                                transition=transition,
                                output_treatment_state=False,
                                output_treatment_cd=False)
        update_units(instructions, stands)

        self.assertEqual(2025, stand1.time)
        self.assertEqual(2025, stand2.time)

        treatment1.assert_has_calls([call(ANY), call(ANY)])
        treatment2.assert_has_calls([call(ANY), call(ANY)])

        transition_fn.assert_has_calls([call(ANY, 2),  # stand1
                                        call(ANY, 1),
                                        call(ANY, 2),
                                        call(ANY, 4),  # stand2
                                        call(ANY, 1),
                                        call(ANY, 2)])
