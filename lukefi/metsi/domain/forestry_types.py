from typing import Sequence

from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.core.condition import Condition
from lukefi.metsi.core.simulation_payload import SimulationPayload

StandList = Sequence[ForestStand]
ForestOpPayload = SimulationPayload[ForestStand]
SimResults = dict[str, list[ForestOpPayload]]
ForestCondition = Condition[ForestStand]
