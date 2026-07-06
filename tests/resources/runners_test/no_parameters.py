from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.app.metsi_control import MetsiControl, AppConfiguration
from lukefi.metsi.core.sim_control import Simulation
from lukefi.metsi.domain.conditions import TimePoints
from lukefi.metsi.core.condition import Condition
from lukefi.metsi.core.instructions import SimulationInstruction
from lukefi.metsi.core.generators import Sequence, Event
from tests.toy_model import ToyModel, ToyTransition, toy_inc


control_structure = MetsiControl[ToyModel](
    app_configuration=AppConfiguration(
        state_format=StateFormat.VMI13,
        run_modes=[RunMode.SIMULATE]
    ),
    simulation=Simulation(instructions=[
        SimulationInstruction(
            # time_points=[1, 2, 3, 4],
            conditions=[TimePoints([1, 2, 3, 4])],
            events=Sequence([
                Event(toy_inc)
            ])
        )
    ],
        transition=ToyTransition(),
        end_condition=Condition[ToyModel](lambda x: x.unit.time > 4)
    ))
