from typing import Optional
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.generators import TreatmentFn
from lukefi.metsi.sim.simulation_payload import SimulationPayload


class MinimumTimeInterval[T: ComputationalUnit](Condition[SimulationPayload[T]]):
    def __init__(self, minimum_time: int, treatment: TreatmentFn[T]) -> None:
        super().__init__(lambda x: _check_eligible_to_run(x, treatment, minimum_time))


def _get_operation_last_run[T: ComputationalUnit](operation_history: list[tuple[int, TreatmentFn[T], dict[str, dict]]],
                                                  operation_tag: TreatmentFn[T]) -> Optional[int]:
    return next((t for t, o, _ in reversed(operation_history) if o == operation_tag), None)


def _check_eligible_to_run[T: ComputationalUnit](
        payload: SimulationPayload[T],
        treatment: TreatmentFn[T],
        minimum_time_interval: int) -> bool:
    last_run = _get_operation_last_run(payload.operation_history, treatment)
    return last_run is None or minimum_time_interval <= (payload.computational_unit.time - last_run)


class TimePoints[T: ComputationalUnit](Condition[SimulationPayload[T]]):
    def __init__(self, time_points: list[int]) -> None:
        super().__init__(lambda x: x.computational_unit.time in time_points)
