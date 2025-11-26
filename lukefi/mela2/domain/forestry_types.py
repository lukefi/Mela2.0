from lukefi.mela2.data.model import ForestStand
from lukefi.mela2.sim.condition import Condition
from lukefi.mela2.sim.simulation_payload import SimulationPayload

StandList = list[ForestStand]
ForestOpPayload = SimulationPayload[ForestStand]
SimResults = dict[str, list[ForestOpPayload]]
ForestCondition = Condition[ForestOpPayload]
