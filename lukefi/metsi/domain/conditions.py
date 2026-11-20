from typing import Optional
from lukefi.metsi.core.condition import Condition
from lukefi.metsi.core.model import ComputationalUnit
from lukefi.metsi.core.simulation_payload import OperationHistory, SimulationPayload
from lukefi.metsi.core.treatment import Treatment


class TimePoints[T: ComputationalUnit](Condition[T]):
    def __init__(self, time_points: list[int]) -> None:
        super().__init__(lambda x: x.unit.time in time_points,
                         "time_points",
                         time_points=set(time_points))


class RelativeTimePoints[T: ComputationalUnit](Condition[T]):
    def __init__(self, relative_time_points: list[int]) -> None:
        super().__init__(lambda x: x.unit.relative_time in relative_time_points,
                         "relative_time_points",
                         relative_time_points=set(relative_time_points))


class TimeSinceTreatment[T: ComputationalUnit](Condition[T]):
    def __init__(self, minimum_time: int, treatment: Treatment[T]) -> None:
        super().__init__(lambda x: _time_since_treatment_at_least(x, treatment, minimum_time), "time_since_treatment")


class TimeSinceTag[T: ComputationalUnit](Condition[T]):
    def __init__(self, minimum_time: int, tag: str) -> None:
        super().__init__(lambda x: _time_since_tag_at_least(x, tag, minimum_time), "time_since_tag")


class MaxTimeSinceTreatment[T: ComputationalUnit](Condition[T]):
    def __init__(self, minimum_time: int, treatment: Treatment[T]) -> None:
        super().__init__(
            lambda x: _time_since_treatment_at_most(x, treatment, minimum_time), "max_time_since_treatment")


class MaxTimeSinceTag[T: ComputationalUnit](Condition[T]):
    def __init__(self, minimum_time: int, tag: str) -> None:
        super().__init__(lambda x: _time_since_tag_at_most(x, tag, minimum_time), "max_time_since_tag")


def _time_since_treatment_at_least[T: ComputationalUnit](
        payload: SimulationPayload[T],
        treatment: Treatment[T],
        minimum_time_interval: int) -> bool:
    last_run = _get_treatment_last_run(payload.operation_history, treatment)
    return last_run is None or minimum_time_interval <= (payload.unit.time - last_run)


def _time_since_treatment_at_most[T: ComputationalUnit](
        payload: SimulationPayload[T],
        treatment: Treatment[T],
        maximum_time_interval: int) -> bool:
    last_run = _get_treatment_last_run(payload.operation_history, treatment)
    return last_run is not None and maximum_time_interval >= (payload.unit.time - last_run)


def _get_treatment_last_run[T: ComputationalUnit](operation_history: OperationHistory,
                                                  treatment: Treatment[T]) -> Optional[int]:
    return next((t for t, o, _, _ in reversed(operation_history) if o == treatment.name), None)


def _time_since_tag_at_least[T: ComputationalUnit](
        payload: SimulationPayload[T],
        tag: str,
        minimum_time_interval: int) -> bool:
    last_run = _get_tag_last_run(payload.operation_history, tag)
    return last_run is None or minimum_time_interval <= (payload.unit.time - last_run)


def _time_since_tag_at_most[T: ComputationalUnit](
        payload: SimulationPayload[T],
        tag: str,
        maximum_time_interval: int) -> bool:
    last_run = _get_tag_last_run(payload.operation_history, tag)
    return last_run is not None and maximum_time_interval >= (payload.unit.time - last_run)


def _get_tag_last_run(operation_history: OperationHistory, tag: str) -> Optional[int]:
    return next((time for time, _, _, tags in reversed(operation_history) if tag in tags), None)
