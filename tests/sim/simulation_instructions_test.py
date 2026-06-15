import sqlite3
import unittest

from lukefi.metsi.domain.utils.file_io import create_database_tables
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
        db = sqlite3.connect(":memory:")
        ToyModel.init_db_tables(db)
        _simulate_unit(payload, config, db)

        cur = db.cursor()
        cur.execute(
            """--sql
                SELECT COUNT(*) FROM nodes WHERE node_type == 3;
            """)
        leaf_count = cur.fetchone()[0]

        self.assertEqual(27, leaf_count)
        cur.execute(
            """--sql
                SELECT identifier FROM nodes WHERE node_type == 3;
            """
        )
        leaf_node_ids_expected = ["0-0-0-0", "0-0-0-1", "0-0-0-2", "0-0-1-0", "0-0-1-1", "0-0-1-2",
             "0-0-2-0", "0-0-2-1", "0-0-2-2", "0-1-0-0", "0-1-0-1", "0-1-0-2",
             "0-1-1-0", "0-1-1-1", "0-1-1-2", "0-1-2-0", "0-1-2-1", "0-1-2-2",
             "0-2-0-0", "0-2-0-1", "0-2-0-2", "0-2-1-0", "0-2-1-1", "0-2-1-2",
             "0-2-2-0", "0-2-2-1", "0-2-2-2",]

        self.assertListEqual(leaf_node_ids_expected, [res[0] for res in cur])

        cur.execute(
            """--sql
                SELECT value FROM toys, nodes WHERE nodes.stand = toys.identifier AND nodes.identifier = toys.node AND nodes.node_type = 3
            """
        )
        self.assertListEqual(
            [3, 4, 5, 4, 5, 6, 5, 6, 7, 4, 5, 6, 5, 6, 7, 6, 7, 8, 5, 6, 7, 6, 7, 8, 7, 8, 9],
            [res[0] for res in cur]
        )
