from lukefi.metsi.domain.conditions import TimePoints
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.generators import Sequence, Event
from tests.toy_model import ToyModel, ToyTransition, toy_inc


control_structure = {
    "simulation_instructions": [
        SimulationInstruction(
            conditions=[TimePoints([1, 2, 3, 4])],
            events=Sequence([
                Event(toy_inc, static_parameters={"incrementation": 2})
            ])
        )
    ],
    "transition": ToyTransition(),
    "end_condition": Condition[ToyModel](lambda x: x.computational_unit.time > 4)
}
