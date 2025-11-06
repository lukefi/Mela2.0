from typing import Any, Optional
import numpy as np
from lukefi.metsi.domain.conditions import MinimumTimeInterval
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.forestry_types import ForestCondition
from lukefi.metsi.sim.generators import Event
from lukefi.metsi.domain.forestry_operations.soil_surface_preparation import soil_surface_preparation
from lukefi.metsi.forestry.harvest.cutting import cutting

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

        # ---- prerequisite (forest_categories) replicated in Python
        def forest_categories(state: ForestStand) -> bool:
            stand = state
            trees = stand.reference_trees

            # stand attributes
            manag_cat = stand.forest_management_category if stand.forest_management_category is not None else -1
            soil_cat = stand.soil_peatland_category.value if stand.soil_peatland_category is not None else -1
            site = stand.site_type_category.value if stand.site_type_category is not None else -1
            year_drain = stand.drainage_year if stand.drainage_year is not None else -1

            # stand structure
            f = trees.stems_per_ha
            d = trees.breast_height_diameter
            h = trees.height

            # basal area (basalArea), stems (N), diameter- and height-weighted means (Dgm, Hgm)
            basal_area = float(np.sum(f * np.pi / 40000.0 * d * d))
            stem_count = float(np.sum(f))
            if basal_area > 0:
                dgm = float(np.sum(f * np.pi / 40000.0 * d * d * d) / basal_area)
                hgm = float(np.sum(f * np.pi / 40000.0 * d * d * h) / basal_area)
            else:
                dgm = 0.0
                hgm = 0.0

            cond_mineral = (
                ((0 <= manag_cat < 3) and soil_cat == 1) or
                ((0 <= manag_cat < 2) and (2 <= soil_cat < 4) and (year_drain <= 1950)) or
                ((0 <= manag_cat < 2) and soil_cat == 2 and (1 <= site < 4) and (1951 <= year_drain)) or
                ((2 <= manag_cat < 3) and (2 <= soil_cat < 5))
            )
            size_ok = (dgm >= 8) and (hgm >= 13.5)
            dense_enough = stem_count > 1.5 * _min_number_of_stems_after_thinning()

            return bool(cond_mineral and size_ok and dense_enough)

        # ---- selection-set eligibility functions
        # Prefer spruce on fertile, pine on poorer (species codes 2=spruce, 1=pine per your proto)
        def _fertility_value(st: ForestStand) -> Optional[int]:
            v = getattr(st.site_type_category, "value", None)
            if v is None:
                return st.site_type_category if isinstance(st.site_type_category, int) else None
            return int(v)

        def prefer_spruce(stand: ForestStand, trees) -> np.ndarray:
            fert = _fertility_value(stand) or 0
            mask = np.zeros(trees.size, dtype=bool)
            if fert <= 3:  # fertile -> prefer spruce
                mask |= (trees.species == 2)
            return mask

        def prefer_pine(stand: ForestStand, trees) -> np.ndarray:
            fert = _fertility_value(stand) or 0
            mask = np.zeros(trees.size, dtype=bool)
            if fert >= 3:  # poorer -> prefer pine
                mask |= (trees.species == 1)
            return mask

        profile_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        profile_y = [0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.25, 0.1, 0.05, 0.05, 0.05]

        tree_selection = {
            "Target": {
                "type": "absolute_remain",
                "var":  "stems_per_ha",
                "amount": _min_number_of_stems_after_thinning(),
            },
            "sets": [
                # Set 1: the “other” species (10% of the remain target), ordered by DBH
                {
                    "sfunction": lambda st, tr: np.logical_not(prefer_spruce(st, tr)) if (_fertility_value(st) or 0) <= 3
                                                else np.logical_not(prefer_pine(st, tr)),
                    "order_var": "breast_height_diameter",
                    "target_var": "stems_per_ha",
                    "target_type": "absolute_remain",
                    "target_amount": 0.1 * _min_number_of_stems_after_thinning(),
                    "profile_x": profile_x,
                    "profile_y": profile_y,
                    "profile_xmode": "relative",
                },
                # Set 2: the preferred species (rest of the target), ordered by DBH
                {
                    "sfunction": prefer_spruce,  # or prefer_pine depending on fertility; handled in sfunction above
                    "order_var": "breast_height_diameter",
                    "target_var": "stems_per_ha",
                    "target_type": "relative",
                    "target_amount": 1.0,
                    "profile_x": profile_x,
                    "profile_y": profile_y,
                    "profile_xmode": "relative",
                },
            ],
            # you can omit "mode" to use cutting() default ("odds_units")
        }

        event_params = {
            "tree_selection": tree_selection,
            "prerequisite": forest_categories,
            "mode": "odds_units",
        } | params

        # Preconditions: 20 years since last cutting. (Add regeneration when you have that treatment.)
        preconds: list[ForestCondition] = [
            MinimumTimeInterval(20, cutting),
        ]

        super().__init__(treatment=cutting, parameters=event_params, preconditions=preconds, **kw)


class Tracks(Event[ForestStand]):
    """Classic Tracks preset: 'even' profile as in your earlier R prototype."""

    def __init__(self, parameters: Optional[dict[str, Any]] = None, **kw) -> None:
        params = parameters or {}

        # --- Eligibility function for this set: here, "all trees are eligible".
        # If you want to exclude species or DBH ranges, do it here.
        def s_all(_stand: ForestStand, trees) -> np.ndarray:
            return np.ones(trees.size, dtype=bool)

        # --- Profile: even selection toward larger DBH (from above) by giving
        #             higher weights to higher order quantiles.
        profile_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        profile_y = [0.00, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 0.90, 1.00, 1.00]

        # --- HARD-CODED tree_selection (user should adjust):
        # Global target: remove 18% of stems_per_ha (relative)
        tree_selection = {
            "Target": {
                "type": "relative",         # "relative" | "absolute" | "absolute_remain"
                "var":  "stems_per_ha",     # frequency variable
                "amount": 0.18,             # remove 18% overall
            },
            "sets": [
                {
                    "sfunction": s_all,
                    "order_var": "breast_height_diameter",  # rank by DBH
                    "target_var": "stems_per_ha",
                    "target_type": "relative",
                    "target_amount": 1.0,                   # full share within this set
                    "profile_x": profile_x,
                    "profile_y": profile_y,
                    "profile_xmode": "relative",            # profile defined over 0..10 classes
                    # "profile_xscale": None,               # optional, leave out unless needed
                }
            ]
        }

        # Required explicit params for strict cutting()
        event_params = {
            "tree_selection": tree_selection,
            "mode": "odds_units",
            # Optional bookkeeping (only set if you want them recorded):
            # "sim_time": 2025,
            # "cutting_method": "MECHANIZED",
        }

        # Allow caller to override anything explicitly
        super().__init__(treatment=cutting, parameters=(event_params | params), **kw)

__all__ = [
    "Mounding",
    "Tracks",
]
