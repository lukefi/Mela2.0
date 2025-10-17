from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.generators import Sequence, Event
from tests.test_utils import inc


control_structure = {
    "simulation_instructions": [
        SimulationInstruction(
            time_points=[1, 2, 3, 4],
            events=Sequence([
                Event(inc)
            ])
        )
    ]
}
