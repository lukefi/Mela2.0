from lukefi.mela2.domain.conditions import TimePoints
from lukefi.mela2.sim.condition import Condition
from lukefi.mela2.sim.simulation_instruction import SimulationInstruction
from lukefi.mela2.sim.generators import Sequence, Event
from tests.toy_model import ToyModel, ToyTransition, toy_inc


control_structure = {
    "simulation_instructions": [
        SimulationInstruction(
            conditions=[TimePoints([1, 2, 3, 4])],
            events=Sequence([
                Event(toy_inc, parameters={"incrementation": 2})
            ])
        )
    ],
    "transition": ToyTransition(),
    "end_condition": Condition[ToyModel](lambda x: x.time > 4)
}
