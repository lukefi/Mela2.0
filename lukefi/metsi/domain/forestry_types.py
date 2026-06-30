from typing import Sequence

from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.simulation_payload import SimulationPayload

StandList = Sequence[ForestStand]
ForestOpPayload = SimulationPayload[ForestStand]
SimResults = dict[str, list[ForestOpPayload]]
ForestCondition = Condition[ForestStand]
