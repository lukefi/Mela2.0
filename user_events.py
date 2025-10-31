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


class Ajourat(Event[ForestStand]):
    """Classic ajourat preset: 'even' profile as in your earlier R prototype."""

    def __init__(self, parameters: Optional[dict[str, Any]] = None, **kw) -> None:
        params = parameters or {}

        # --- Eligibility function for this set: here, "all trees are eligible".
        # If you want to exclude species or DBH ranges, do it here.
        def s_all(_stand: ForestStand, trees) -> np.ndarray:
            return np.ones(trees.size, dtype=bool)

        # --- Profile: bias selection toward larger DBH (from above) by giving
        #             higher weights to higher order quantiles.
        profile_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        profile_y = [0.00, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 0.90, 1.00, 1.00]

        # --- HARD-CODED tree_selection (user should adjust):
        # Global target: remove 25% of stems_per_ha (relative)
        tree_selection = {
            "Target": {
                "type": "relative",         # "relative" | "absolute" | "absolute_remain"
                "var":  "stems_per_ha",     # frequency variable
                "amount": 0.25,             # remove 25% overall
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
            ],
            "mode": "odds_units"
        }

        # Required explicit params for strict cutting()
        event_params = {
            "tree_selection": tree_selection,
            "freq_var": "stems_per_ha",       # REQUIRED by cutting()
            "select_from_all": False,         # REQUIRED by cutting() (bool)
            # Optional bookkeeping (only set if you want them recorded):
            # "sim_time": 2025,
            # "cutting_method": "MECHANIZED",
            # "mode": "odds_units",
        }

        # Allow caller to override anything explicitly
        super().__init__(treatment=cutting, parameters=(event_params | params), **kw)

__all__ = [
    "Mounding",
    "Ajourat",
]
