import unittest

from lukefi.metsi.domain.conditions import TimeSinceTreatment
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.generators import Event, Sequence
from lukefi.metsi.sim.sim_configuration import SimConfiguration
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from tests.toy_model import ToyModel, ToyTransition


class SimConfigurationTest(unittest.TestCase):
    def test_sim_configuration(self):
        def fn1(x):
            return x

        def fn2(y):
            return y

        config = {
            'simulation_instructions': [
                SimulationInstruction(
                    # time_points=[1, 2, 3],
                    events=[
                        Event(
                            preconditions=[
                                TimeSinceTreatment(5, fn1)
                            ],
                            treatment=fn1,
                            parameters={
                                'param1': 1
                            }
                        ),
                        Event(
                            treatment=fn2
                        )
                    ]
                ),
                SimulationInstruction(
                    # time_points=[3, 4, 5],
                    events=[
                        Sequence([
                            Event(
                                preconditions=[
                                    TimeSinceTreatment(5, fn1)
                                ],
                                treatment=fn1,
                                parameters={
                                    'param1': 1
                                }
                            ),
                            Event(
                                treatment=fn2
                            )
                        ])
                    ]
                ),
            ],
            "transition": ToyTransition(),
            "end_condition": Condition[ToyModel](lambda x: x.time > 5)
        }
        result = SimConfiguration(**config)
        self.assertEqual(2, len(result.instructions))
