from typing import Any, Optional
import numpy as np
from lukefi.metsi.domain.conditions import MinimumTimeInterval
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.sim.condition import Condition
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.generators import Event
from lukefi.metsi.forestry.harvest.cutting import cutting
from lukefi.metsi.domain.forestry_treatments.soil_surface_preparation import soil_surface_preparation
from lukefi.metsi.domain.forestry_treatments.regeneration import regeneration
from lukefi.metsi.domain.collected_data import RemovedTrees
from lukefi.metsi.data.enums.mela import MelaMethodOfTheLastCutting

def _forest_categories_check(_time: int, payload: SimulationPayload[ForestStand]) -> bool:
    stand = payload.computational_unit
    stand.update_aggregates()  # use stand aggregates, not manual BA math

    manag_cat = stand.forest_management_category if stand.forest_management_category is not None else -1
    soil_cat = stand.soil_peatland_category.value if stand.soil_peatland_category is not None else -1
    site = stand.site_type_category.value if stand.site_type_category is not None else -1
    year_drain = stand.drainage_year if stand.drainage_year is not None else -1

    stem_count = float(stand.stems_per_ha or 0.0)
    dgm = float(stand.weighted_mean_diameter or 0.0)
    hgm = float(stand.weighted_mean_height or 0.0)

    cond_mineral = (
        ((0 <= manag_cat < 3) and soil_cat == 1) or
        ((0 <= manag_cat < 2) and (2 <= soil_cat < 4) and (year_drain <= 1950)) or
        ((0 <= manag_cat < 2) and soil_cat == 2 and (1 <= site < 4) and (1951 <= year_drain)) or
        ((2 <= manag_cat < 3) and (2 <= soil_cat < 5))
    )
    size_ok = (dgm >= 8) and (hgm >= 13.5)
    dense_enough = stem_count > 1.5 * 1000

    return bool(cond_mineral and size_ok and dense_enough)
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

class FirstThinningMineralSoils(Event[ForestStand]):
    """
    First thinning on mineral soils, ported from the R prototype in
    event_first_thinning_example.txt.

    Defaults:
      - Keep 1000 stems/ha (absolute_remain on stems_per_ha)
      - Profile favors removing larger DBH classes (as in the proto)
      - Two selection sets to bias species by site fertility
      - Requires at least 20 years since last cutting
    """

    def __init__(self, parameters: Optional[dict[str, Any]] = None, **kw) -> None:
        params = parameters or {}

        # ---- helper: minimum stems after thinning
        def _min_number_of_stems_after_thinning() -> int:
            return 1000  # default per the example file

        def s_conifer_bias(stand: ForestStand, trees) -> np.ndarray:
            fert = (stand.site_type_category or 0)
            if fert == 3:
                # include both spruce (2) and pine (1)
                return (trees.species == 1) | (trees.species == 2)
            if fert < 3:
                # fertile: prefer spruce
                return trees.species == 2

            # Otherwise prefer pine
            return trees.species == 1

        profile_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        profile_y = [0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.25, 0.1, 0.05, 0.05, 0.05]

        tree_selection = {
            "target": {
                "type": "absolute_remain",
                "var":  "stems_per_ha",
                "amount": _min_number_of_stems_after_thinning(),
            },
            "sets": [
                {
                    "sfunction": s_conifer_bias,
                    "order_var": "breast_height_diameter",
                    "target_var": "stems_per_ha",
                    "target_type": "absolute_remain",
                    "target_amount": 0.1 * _min_number_of_stems_after_thinning(),
                    "profile_x": profile_x,
                    "profile_y": profile_y,
                    "profile_xmode": "relative",
                },
                {
                    "sfunction": s_conifer_bias,
                    "order_var": "breast_height_diameter",
                    "target_var": "stems_per_ha",
                    "target_type": "relative",
                    "target_amount": 1.0,
                    "profile_x": profile_x,
                    "profile_y": profile_y,
                    "profile_xmode": "relative",
                },
            ],
        }

        event_params = {
            "tree_selection": tree_selection,
            "mode": "odds_units",
            "cutting_method": MelaMethodOfTheLastCutting.FIRST_THINNING.value,
        } | params

        # --- Preconditions now include both: 20y spacing AND forest_categories
        preconds: list[Condition[SimulationPayload[ForestStand]]] = [
            MinimumTimeInterval(20, cutting),
            Condition(_forest_categories_check),
        ]

        super().__init__(treatment=cutting, parameters=event_params,
                        preconditions=preconds,
                        collected_data={RemovedTrees},
                        **kw)


class Tracks(Event[ForestStand]):
    """Classic Tracks preset: 'even' profile as in your earlier R prototype."""

    def __init__(self,
                 parameters: Optional[dict[str, Any]] = None,
                 preconditions: Optional[list[Condition[SimulationPayload[ForestStand]]]] = None,
                 postconditions: Optional[list[Condition[SimulationPayload[ForestStand]]]] = None,
                 file_parameters: Optional[dict[str, str]] = None,
                 **kw) -> None:
        params = parameters or {}

        def s_all(_stand: ForestStand, trees) -> np.ndarray:
            return np.ones(trees.size, dtype=bool)

        profile_x = [0,1]
        profile_y = [0.18,0.18]

        tree_selection = {
            "target": {"type": "relative", "var": "stems_per_ha", "amount": 0.18},
            "sets": [{
                "sfunction": s_all,
                "order_var": "breast_height_diameter",
                "target_var": "stems_per_ha",
                "target_type": "relative",
                "target_amount": 1.0,
                "profile_x": profile_x,
                "profile_y": profile_y,
                "profile_xmode": "relative",
            }],
        }

        event_params = {
            "tree_selection": tree_selection,
            "mode": "odds_units",
            "cutting_method": MelaMethodOfTheLastCutting.FIRST_THINNING.value,
        } | params

        # Default: at least 20y since last cutting and forest category check
        default_preconds: list[Condition[SimulationPayload[ForestStand]]] = [
            MinimumTimeInterval(20, cutting),
            Condition(_forest_categories_check),
        ]

        super().__init__(
            treatment=cutting,
            parameters=event_params,
            preconditions=default_preconds + (preconditions or []),
            postconditions=postconditions,
            file_parameters=file_parameters,
            collected_data={RemovedTrees},
            **kw
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



__all__ = [
    "Mounding",
    "Tracks",
    "FirstThinningMineralSoils",
    "PlantingPines"
]
