from lukefi.metsi.domain.conditions import TimeSinceTreatment, TimePoints
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.simulation_instruction import SimulationInstruction
from lukefi.metsi.sim.generators import Alternatives, Sequence, Event
from lukefi.metsi.sim.operations import do_nothing
from tests.toy_model import ToyModel, ToyTransition, toy_inc


control_structure = {
    "simulation_instructions": [
        SimulationInstruction(
            # time_points=[1, 2, 3, 4],
            conditions=[TimePoints([1, 2, 3, 4])],
            events=Sequence([
                Sequence([
                    Event(do_nothing),
                ]),
                Alternatives([
                    Event(do_nothing),
                    Event(
                        preconditions=[
                            TimeSinceTreatment(2, toy_inc)
                        ],
                        treatment=toy_inc
                    )
                ])
            ])
        )
    ],
    "transition": ToyTransition(),
    "end_condition": Condition[ToyModel](lambda x: x.computational_unit.time > 4)
}
