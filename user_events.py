from typing import Any, Optional
import numpy as np
from lukefi.metsi.domain.conditions import MinimumTimeInterval
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.sim.generators import Event
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.domain.forestry_treatments.mark_trees import mark_trees
from lukefi.metsi.domain.forestry_treatments.soil_surface_preparation import soil_surface_preparation
from lukefi.metsi.domain.forestry_treatments.regeneration import regeneration


def _min_regeneration_diameter(stand: ForestStand) -> float:

    if stand.site_type_category in (1, 2):
        return 28.0
    elif stand.site_type_category == 3:
        return 26.0
    elif stand.site_type_category == 4:
        return 25.0
    # site >= 5 or unknown
    return 22.0


def _forest_categories_regeneration(time_point: int, payload: Any) -> bool:

    _ = time_point  # not used
    stand: ForestStand = payload.computational_unit  # SimulationPayload[ForestStand]

    # Map R variables to Python model fields
    manag_cat = stand.forest_management_category
    site_idx = stand.site_type_category
    dgm = stand.weighted_mean_diameter

    # If any required value is missing, the condition cannot be satisfied
    if manag_cat is None or site_idx is None or dgm is None:
        return False

    dmin = _min_regeneration_diameter(stand)

    return (
        manag_cat < 2
        and site_idx < 5
        and dgm > 0.95 * dmin
    )


class MarkRetentionTrees(Event[ForestStand]):
    """
    Example event for selecting retention trees.

      - Prerequisite (stand-level condition) `forest_categories_regeneration`:
          * management category < 2
          * site type < 5
          * Dgm > 0.95 * MIN_REGENERATION_DIAMETER(stand)

      - Treatment:
          * selects 10 stems/ha (absolute target on stems_per_ha)
          * uses three selection sets:
              1) trees older than 60 years (breast_height_age > 60)
              2) other species than pine, spruce or birches (species code > 4)
              3) trees with diameter > 15 cm
          * marked trees get attributes like:
              tree_type = "SPARE"
              management_category = 2
    """

    def __init__(
        self,
        parameters: Optional[dict[str, Any]] = None,
        preconditions: Optional[list[ForestCondition]] = None,
        postconditions: Optional[list[ForestCondition]] = None,
        file_parameters: Optional[dict[str, str]] = None,
    ) -> None:
        params = parameters or {}

        # trees older than 60 years
        def s_age_gt_60(_stand: ForestStand, trees) -> np.ndarray:
            return trees.breast_height_age > 60

        # other species than pine, spruce or birches
        def s_other_species(_stand: ForestStand, trees) -> np.ndarray:
            return trees.species > 4

        # trees with diameter > 15 cm
        def s_large_diameter(_stand: ForestStand, trees) -> np.ndarray:
            return trees.breast_height_diameter > 15

        tree_selection = {
            "Target": {
                "type": "absolute",
                "var": "stems_per_ha",
                "amount": 10.0,  # 10 stems/ha
            },
            "sets": [
                {
                    "sfunction": s_age_gt_60,
                    "order_var": "breast_height_age",
                    "target_var": "stems_per_ha",
                    "target_type": "relative",
                    "target_amount": 1.0,
                    "profile_x": [0.0, 1.0],
                    "profile_y": [0.01, 0.999],
                    "profile_xmode": "relative",
                },
                {
                    "sfunction": s_other_species,
                    "order_var": "breast_height_diameter",
                    "target_var": "stems_per_ha",
                    "target_type": "relative",
                    "target_amount": 0.7,
                    "profile_x": [0.0, 0.5, 1.0],
                    "profile_y": [0.01, 0.05, 0.999],
                    "profile_xmode": "relative",
                },
                {
                    "sfunction": s_large_diameter,
                    "order_var": "breast_height_diameter",
                    "target_var": "stems_per_ha",
                    "target_type": "relative",
                    "target_amount": 0.2,
                    "profile_x": [0.0, 0.5, 1.0],
                    "profile_y": [0.01, 0.05, 0.999],
                    "profile_xmode": "relative",
                },
            ],
        }

        default_params: dict[str, Any] = {
            "tree_selection": tree_selection,
            "select_from_all": True,
            "mode": "odds_units",
            "attributes": {
                "tree_type": "SPARE",
                "management_category": 2,
            },
            "labels": ["retention_trees"],
        }

        merged_params = default_params | params

        # --- prerequisite: forest_categories_regeneration ---
        default_preconds: list[ForestCondition] = [  # type: ignore[list-item]
            Condition(_forest_categories_regeneration)
        ]
        merged_preconds = default_preconds + (preconditions or [])

        super().__init__(
            treatment=mark_trees,
            parameters=merged_params,
            preconditions=merged_preconds,
            postconditions=postconditions,
            file_parameters=file_parameters,
        )
class PlantingPines(Event[ForestStand]):
    """
    Pine planting event that calls regeneration with sensible defaults.
    Override by passing 'parameters={...}' when constructing, or subclass for species presets.
    """
    def __init__(self,
                 parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[ForestCondition]] = None,
                 postconditions: Optional[list[ForestCondition]] = None,
                 file_parameters: Optional[dict[str, str]] = None) -> None:

        default_params: dict[str, Any] = {
            "origin": 2,           # planted
            "method": 2,
            "species": 1,          # Pine
            "stems_per_ha": 1500.0,
            "height": 0.7,
            "biological_age": 3.0,
            "type": "artificial",
        }

        merged = default_params | (parameters or {})
        super().__init__(treatment=regeneration,
                         parameters=merged,
                         preconditions=preconditions,
                         postconditions=postconditions,
                         file_parameters=file_parameters)


class Mounding(Event[ForestStand]):
    """
    Mounding Event using soil surface preparation..

    Wraps the `soil_surface_preparation` treatment with sensible defaults
    and a time-spacing precondition. By default, it sets:
      - method="mounding"
      - intensity=1200.0 (per ha)
      - labels=["ssp_default"]

    A minimum 20-year interval since the last soil surface preparation treatment.

    Parameters
    ----------
    parameters : dict[str, Any] | None
        Optional overrides for treatment parameters (e.g. "method", "intensity",
        "labels"). Merged over the defaults above.
    preconditions : list[ForestCondition] | None
        Additional preconditions; appended to the default 20-year interval rule.
    postconditions : list[ForestCondition] | None
        Optional postconditions evaluated after the event is simulated.
    file_parameters : dict[str, str] | None
        Optional parameters loaded from files.

    """
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
            "labels": ["mounding"],
        }
        # Default preconditions: at least 20 years since this treatment last ran
        default_preconds: list[ForestCondition] = [
            MinimumTimeInterval(20, soil_surface_preparation)
        ]

        merged_params = defaults | (parameters or {})
        merged_preconds = default_preconds + (preconditions or [])

        super().__init__(
            treatment=soil_surface_preparation,
            parameters=merged_params,
            preconditions=merged_preconds,
            postconditions=postconditions,
            file_parameters=file_parameters,
        )

__all__ = [
    "Mounding",
    "MarkRetentionTrees",
    "PlantingPines",
]
