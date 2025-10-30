from typing import Any, Optional
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta
from lukefi.metsi.domain.natural_processes.grow_metsi import grow_metsi
from lukefi.metsi.domain.natural_processes.grow_motti_dll import grow_motti_dll
from lukefi.metsi.sim.generators import Event
from lukefi.metsi.sim.operations import do_nothing
from lukefi.metsi.forestry.harvest.cutting import cutting

class DoNothing(Event[ForestStand]):
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(treatment=do_nothing,
                         parameters=parameters,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)


class GrowActa(Event[ForestStand]):
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(treatment=grow_acta,
                         parameters=parameters,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)


class GrowMetsi(Event[ForestStand]):
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(treatment=grow_metsi,
                         parameters=parameters,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)


class GrowMotti(Event[ForestStand]):
    def __init__(self,
                 parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(treatment=grow_motti_dll,
                         parameters=parameters,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)


class Ajourat(Event[ForestStand]):
    """Classic ajourat preset: 'even' profile as in your earlier R prototype."""
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        defaults = {
            "proportion": 0.30,
            "order_var": "breast_height_diameter",
            "profile": "flat",  # like earlier ajourat
            "target_var": "stems_per_ha",
            "target_type": "relative",
        }
        super().__init__(treatment=cutting, parameters=(defaults | (parameters or {})),
                         preconditions=preconditions, postconditions=postconditions, file_parameters=file_parameters)


__all__ = [
    "DoNothing",
    "GrowMotti",
    "GrowActa",
    "GrowMetsi",
    "Ajourat",
]
