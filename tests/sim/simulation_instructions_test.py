import unittest

from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.generators import Alternatives, Event
from lukefi.metsi.sim.sim_configuration import SimConfiguration
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.simulator import _simulate_unit
from tests.toy_model import ToyModel, ToyTransition, toy_inc


class SimulationInstructionsTest(unittest.TestCase):

    def test_multiple_instructions(self):
        declaration = {
            "simulation_instructions": [
                SimulationInstruction(
                    events=[
                        Event(toy_inc, static_parameters={
                            "incrementation": 1
                        })
                    ]
                ),
                SimulationInstruction(
                    events=[
                        Alternatives([
                            Event(toy_inc, static_parameters={
                                "incrementation": 2
                            }),
                            Event(toy_inc, static_parameters={
                                "incrementation": 3
                            }),
                        ])
                    ]
                )
            ],
            "transition": ToyTransition(),
            "end_condition": Condition[ToyModel](lambda payload: payload.computational_unit.time >= 3)
        }

        config = SimConfiguration[ToyModel](**declaration)
        payload = SimulationPayload[ToyModel](computational_unit=ToyModel("test", 0))

        results = _simulate_unit(payload, config)

        self.assertEqual(27, len(results))
        self.assertListEqual(
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 2],
             [0, 0, 2, 0], [0, 0, 2, 1], [0, 0, 2, 2], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 0, 2],
             [0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 1, 2], [0, 1, 2, 0], [0, 1, 2, 1], [0, 1, 2, 2],
             [0, 2, 0, 0], [0, 2, 0, 1], [0, 2, 0, 2], [0, 2, 1, 0], [0, 2, 1, 1], [0, 2, 1, 2],
             [0, 2, 2, 0], [0, 2, 2, 1], [0, 2, 2, 2],],
            [result.node_id for result in results]
        )
        self.assertListEqual(
            [3, 4, 5, 4, 5, 6, 5, 6, 7, 4, 5, 6, 5, 6, 7, 6, 7, 8, 5, 6, 7, 6, 7, 8, 7, 8, 9],
            [result.computational_unit.value for result in results]
        )
