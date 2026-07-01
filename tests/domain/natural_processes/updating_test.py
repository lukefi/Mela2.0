from unittest.mock import ANY
from unittest.mock import call
from unittest.mock import Mock
import unittest
from lukefi.metsi.domain.natural_processes.updating import update_to_year_fn


class TestUpdating(unittest.TestCase):

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
