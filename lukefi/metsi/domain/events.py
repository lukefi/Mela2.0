from typing import Any, Optional
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.collected_data import NaturalProcessInfo
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta_fn
from lukefi.metsi.domain.natural_processes.grow_metsi import grow_metsi_fn
from lukefi.metsi.domain.natural_processes.grow_motti_dll import grow_motti_dll_fn
from lukefi.metsi.core.generators import Event
from lukefi.metsi.core.treatment import Treatment, do_nothing


class DoNothing(Event[ForestStand]):
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(treatment=do_nothing,
                         static_parameters=parameters,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)


class GrowActa(Event[ForestStand]):
    def __init__(self,
                 max_step: int = 5,
                 parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(
            treatment=Treatment(
                lambda state: grow_acta_fn(
                    state,
                    max_step),
                "grow_acta",
                collected_data={NaturalProcessInfo}),
            static_parameters=parameters,
            preconditions=preconditions,
            postconditions=postconditions,
            file_parameters=file_parameters)


class GrowMetsi(Event[ForestStand]):
    def __init__(self,
                 max_step: int = 5,
                 parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(
            treatment=Treatment(
                lambda state: grow_metsi_fn(
                    state,
                    max_step),
                "grow_metsi",
                collected_data={NaturalProcessInfo}),
            static_parameters=parameters,
            preconditions=preconditions,
            postconditions=postconditions,
            file_parameters=file_parameters)


class GrowMotti(Event[ForestStand]):
    def __init__(self,
                 max_step: int = 5,
                 parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(
            treatment=Treatment(
                lambda state, **params: grow_motti_dll_fn(
                    state,
                    max_step,
                    **params),
                "grow_motti_dll"),
            static_parameters=parameters,
            preconditions=preconditions,
            postconditions=postconditions,
            file_parameters=file_parameters)
