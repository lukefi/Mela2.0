from typing import Any, Optional
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta
from lukefi.metsi.domain.natural_processes.grow_metsi import grow_metsi
from lukefi.metsi.domain.natural_processes.grow_motti_dll import grow_motti_dll
from lukefi.metsi.sim.generators import Event
from lukefi.metsi.sim.operations import do_nothing
from lukefi.metsi.forestry.harvest.ftrt_cutting import cutting

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




class FirstThinning(Event[ForestStand]):
    """Typical first thinning: from below, modest removal, optionally species-filtered."""
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:

        defaults = {
            "proportion": 0.25,                 # remove 25% of stems among selected trees
            "order_var": "breast_height_diameter",
            "profile": "below",                 # thinning from below
            "target_var": "stems_per_ha",
            "target_type": "relative",
            # Optional selectors:
            # "species": [1],                   # e.g., pine only
            # "dbh_min": 6.0, "dbh_max": 18.0,
        }
        super().__init__(treatment=cutting,
                         parameters=(defaults | (parameters or {})),
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)


class ThinningFromBelow(Event[ForestStand]):
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        defaults = {
            "proportion": 0.30,
            "order_var": "breast_height_diameter",
            "profile": "below",
            "target_var": "stems_per_ha",
            "target_type": "relative",
        }
        super().__init__(treatment=cutting, parameters=(defaults | (parameters or {})),
                         preconditions=preconditions, postconditions=postconditions, file_parameters=file_parameters)


class ThinningFromAbove(Event[ForestStand]):
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        defaults = {
            "proportion": 0.20,
            "order_var": "breast_height_diameter",
            "profile": "above",
            "target_var": "stems_per_ha",
            "target_type": "relative",
        }
        super().__init__(treatment=cutting, parameters=(defaults | (parameters or {})),
                         preconditions=preconditions, postconditions=postconditions, file_parameters=file_parameters)


class EvenThinning(Event[ForestStand]):
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        defaults = {
            "proportion": 0.30,
            "order_var": "breast_height_diameter",
            "profile": "flat",  # synonymous with "even"
            "target_var": "stems_per_ha",
            "target_type": "relative",
        }
        super().__init__(treatment=cutting, parameters=(defaults | (parameters or {})),
                         preconditions=preconditions, postconditions=postconditions, file_parameters=file_parameters)


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
]
