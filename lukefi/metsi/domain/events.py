from typing import Any, Optional
from lukefi.metsi.domain.conditions import MinimumTimeInterval
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta
from lukefi.metsi.domain.natural_processes.grow_metsi import grow_metsi
from lukefi.metsi.domain.natural_processes.grow_motti_dll import grow_motti_dll
from lukefi.metsi.sim.generators import Event
from lukefi.metsi.sim.operations import do_nothing
from lukefi.metsi.domain.forestry_operations.ftrt_soil_surface_preparation import ftrt_soil_surface_preparation


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



class SoilSurfacePreparation(Event[ForestStand]):
    """Store soil surface preparation metadata and enforce a sensible default interval."""
    def __init__(
        self,
        parameters: Optional[dict[str, Any]] = None,
        preconditions: Optional[list[ForestCondition]] = None,
        postconditions: Optional[list[ForestCondition]] = None,
        file_parameters: Optional[dict[str, str]] = None,
    ) -> None:
        defaults = {
            "method": "mounding",
            "intensity": 1200.0,
            "labels": ["ssp_default"],
        }
        # Default preconditions: at least 20 years since this treatment last ran
        default_preconds: list[ForestCondition] = [
            MinimumTimeInterval(20, ftrt_soil_surface_preparation)
        ]

        merged_params = defaults | (parameters or {})
        merged_preconds = default_preconds + (preconditions or [])

        super().__init__(
            treatment=ftrt_soil_surface_preparation,
            parameters=merged_params,
            preconditions=merged_preconds,
            postconditions=postconditions,
            file_parameters=file_parameters,
        )


__all__ = [
    "DoNothing",
    "GrowMotti",
    "GrowActa",
    "GrowMetsi",
    "SoilSurfacePreparation",
]
