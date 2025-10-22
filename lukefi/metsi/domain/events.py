from typing import Any, Optional
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta
from lukefi.metsi.domain.natural_processes.grow_metsi import grow_metsi
from lukefi.metsi.sim.generators import Event
from lukefi.metsi.sim.operations import do_nothing
from lukefi.metsi.forestry.naturalprocess.ftrt_regeneration import ftrt_regeneration


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


class Planting(Event[ForestStand]):
    """
    Base planting event that calls regeneration with sensible defaults.
    Override by passing 'parameters={...}' when constructing, or subclass for species presets.
    """
    def __init__(self,
                 parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:

        # Defaults aligned with event_planting_example/test_planting
        default_params: dict[str, Any] = {
            "origin": 2,           # planted
            "method": 2,           # manual (accepted but not used by treatment)
            "species": 3,          # example used silver birch
            "stems_per_ha": 1500.0,
            "height": 0.7,
            "biological_age": 3.0,
            # You may also set ntrees (default 10), breast_height_diameter/age if desired
            # "ntrees": 10,
        }

        merged = default_params | (parameters or {})
        super().__init__(treatment=ftrt_regeneration,
                         parameters=merged,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)


class PlantingPines(Planting):
    """
    Convenience event: 'Planting' + pine-leaning defaults.
    You can still override anything via parameters=...
    """
    def __init__(self,
                 parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:

        pine_defaults = {
            "species": 1,          # typical code for Scots pine in many schemas
            # Optionally adjust stocking/height if you like; leaving only species here:
            # "stems_per_ha": 1800.0,
            # "height": 0.5,
        }

        merged = pine_defaults | (parameters or {})
        super().__init__(parameters=merged,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)

__all__ = [
    "DoNothing",
    "GrowActa",
    "GrowMetsi",
    "Planting",
    "PlantingPines",
]
