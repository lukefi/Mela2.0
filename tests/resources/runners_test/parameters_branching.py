from lukefi.metsi.domain.conditions import TimePoints
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.generators import Alternatives, Sequence, Event
from lukefi.metsi.sim.treatment import do_nothing
from tests.toy_model import ToyModel, ToyTransition, toy_inc


control_structure = {
    "simulation_instructions": [
        SimulationInstruction(
            conditions=[TimePoints([1, 2])],
            events=Sequence([
                Sequence([Event(do_nothing)]),
                Alternatives([
                    Event(do_nothing),
                    Event(toy_inc, static_parameters={"incrementation": 1}),
                    Event(toy_inc, static_parameters={"incrementation": 2})
                ])
            ])
        )
    ],
    "transition": ToyTransition(),
    "end_condition": Condition[ToyModel](lambda x: x.computational_unit.time > 2)
}
