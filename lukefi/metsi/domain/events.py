from typing import Any, Optional
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.domain.natural_processes.grow_acta import grow_acta
from lukefi.metsi.domain.natural_processes.grow_metsi import grow_metsi
from lukefi.metsi.domain.natural_processes.grow_motti_dll import grow_motti_dll
from lukefi.metsi.sim.generators import Event
from lukefi.metsi.sim.operations import do_nothing

# --- New treatments (atomic ops)
from lukefi.metsi.domain.forestry_operations.ftrt_cutting import ftrt_cutting
from lukefi.metsi.domain.forestry_operations.ftrt_soil_surface_preparation import ftrt_soil_surface_preparation
from lukefi.metsi.domain.natural_processes.ftrt_regeneration import ftrt_regeneration
from lukefi.metsi.domain.forestry_operations.ftrt_mark_trees import ftrt_mark_trees

# --- Composite “event flows”
from lukefi.metsi.domain.forestry_operations.events_regeneration import event_regeneration_chain
from lukefi.metsi.domain.forestry_operations.events_r_ported import (
    event_first_thinning_ajourat,
    event_ba_thinning_from_below_regular_coniferPriority,
    event_ba_thinning_from_below_regular_coniferPriority2,
    event_ba_thinning_from_below_regular_coniferPriority3,
    event_ba_thinning_from_below_regular_coniferBirchPriority,
)

# --- Optional: thinning cores if you want wrappers for them too
from lukefi.metsi.domain.forestry_operations.thin_basal_area import ftrt_thin_basal_area
from lukefi.metsi.domain.forestry_operations.thin_nr_of_stems import ftrt_thin_nr_of_stems




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



class Cutting(Event[ForestStand]):
    """Generic cutting using a caller-defined selection profile."""
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(treatment=ftrt_cutting,
                         parameters=parameters,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)

class SoilSurfacePreparation(Event[ForestStand]):
    """Store soil surface prep info and log it."""
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(treatment=ftrt_soil_surface_preparation,
                         parameters=parameters,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)

class Regeneration(Event[ForestStand]):
    """Add a new stratum for planting/regeneration."""
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(treatment=ftrt_regeneration,
                         parameters=parameters,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)

class MarkTrees(Event[ForestStand]):
    """Clone & tag selected trees (e.g., retention), conserving stems."""
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(treatment=ftrt_mark_trees,
                         parameters=parameters,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)

class ThinBasalArea(Event[ForestStand]):
    """Basal-area thinning core (from-below by default)."""
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(treatment=ftrt_thin_basal_area,
                         parameters=parameters,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)

class ThinNumberOfStems(Event[ForestStand]):
    """First-thinning style ‘keep at least N stems’ selection."""
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(treatment=ftrt_thin_nr_of_stems,
                         parameters=parameters,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)

# ------------ Composite “event chains” as Events ------------

class RegenerationChain(Event[ForestStand]):
    """Mark retention -> clearcut -> soil prep -> planting (one call)."""
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(treatment=event_regeneration_chain,
                         parameters=parameters,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)

class FirstThinningAjourat(Event[ForestStand]):
    """Ajourat (18%) + first thinning to instruction-limited stems."""
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(treatment=event_first_thinning_ajourat,
                         parameters=parameters,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)

class BAThinningConiferPriority(Event[ForestStand]):
    """BA-thinning from below; prefer conifers (variant 1)."""
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(treatment=event_ba_thinning_from_below_regular_coniferPriority,
                         parameters=parameters,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)

class BAThinningConiferPriority2(Event[ForestStand]):
    """BA-thinning; keep ≥2 m²/ha of ‘other species’."""
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(treatment=event_ba_thinning_from_below_regular_coniferPriority2,
                         parameters=parameters,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)

class BAThinningConiferPriority3(Event[ForestStand]):
    """BA-thinning; explicit cap on other-species removal (e.g. 0.7)."""
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(treatment=event_ba_thinning_from_below_regular_coniferPriority3,
                         parameters=parameters,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)

class BAThinningBirchPriority(Event[ForestStand]):
    """BA-thinning; prioritize birches first, conifers second."""
    def __init__(self, parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:
        super().__init__(treatment=event_ba_thinning_from_below_regular_coniferBirchPriority,
                         parameters=parameters,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)


__all__ = [
    "DoNothing",
    "GrowMotti",
    "GrowActa",
    "GrowMetsi",

    # Atomic
    "Cutting",
    "SoilSurfacePreparation",
    "Regeneration",
    "MarkTrees",
    "ThinBasalArea",
    "ThinNumberOfStems",

    # Composites
    "RegenerationChain",
    "FirstThinningAjourat",
    "BAThinningConiferPriority",
    "BAThinningConiferPriority2",
    "BAThinningConiferPriority3",
    "BAThinningBirchPriority",
]
