from __future__ import annotations
from typing import Any, Optional, Callable
import numpy as np

from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.domain.forestry_operations.metrics.stand_metrics import compute_stand_metrics

# Compute other-species G share to convert "remain 2 m2/ha" into a relative cap
from lukefi.metsi.domain.forestry_operations.metrics.selection_data import SelectionData
# Ported treatments
from lukefi.metsi.domain.forestry_operations.ajourat import ftrt_ajourat
from lukefi.metsi.domain.forestry_operations.thin_nr_of_stems import ftrt_thin_nr_of_stems
from lukefi.metsi.domain.forestry_operations.thin_basal_area import ftrt_thin_basal_area

from lukefi.metsi.data.limit_tables.limit_constants import (
    basal_area_instruction_lower_limit,
    basal_area_instructions_upper_limit,
    min_number_of_stems_after_thinning,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _fertility(stand: ForestStand) -> int:
    """
    Map R 'fertility' (site class) to a stand attribute.
    Uses stand.site if present, else stand.site_type_category, else 3 as a neutral default.
    """
    if hasattr(stand, "site") and stand.site is not None:
        return int(stand.site)
    if hasattr(stand, "site_type_category") and stand.site_type_category is not None:
        return int(stand.site_type_category)
    return 3

def _sfunctions_conifer_priority(stand: ForestStand) -> tuple[Callable, Callable]:
    """
    Build the same pair of sfunctions used in R:
    - sfunction2: prefer conifer that suits the site (spruce on fertile, pine on poorer)
    - sfunction1: complement of sfunction2 (the others)
    Species codes assumed: 1=pine, 2=spruce, 3/4=birches. (Matches model enums.)
    """
    fert = _fertility(stand)

    def sfunction2(ctx: ForestStand, data: ReferenceTrees):
        # prefer spruce on fertile (<=3), pine on poorer (>=3)
        if fert <= 3:
            return data["species"] == 2
        return data["species"] == 1

    def sfunction1(ctx: ForestStand, data: ReferenceTrees):
        return ~sfunction2(ctx, data)

    return sfunction1, sfunction2

# ----------------------------------------------------------------------
# Event: Ensiharvennus + Ajourat (first thinning with strip roads)
# R reference: event_ensiharvennus_ajourat.txt
# ----------------------------------------------------------------------

def event_first_thinning_ajourat(
    op: OpTuple[ForestStand],
    *,
    sim_time: Optional[int] = None
) -> OpTuple[ForestStand]:
    """
    R flow:
      1) Ajourat: remove 18% of stems evenly.
      2) First thinning by stems: cap overall removal to (0.5-0.18)/(1-0.18),
         and target MIN_NUMBER_OF_STEMS_AFTER_THINNING(stand).
    """
    stand, cdata = op

    # 1) Ajourat (18%)
    (stand, cdata) = ftrt_ajourat((stand, cdata), proportion=0.18, labels=["ajourat"], sim_time=sim_time)

    # 2) First thinning by number of stems:
    max_prop = (0.5 - 0.18) / (1 - 0.18)   # “including ajourat, at most half of stems removed” (from R)
    stems_after = min_number_of_stems_after_thinning(stand, default=1500.0)

    (stand, cdata) = ftrt_thin_nr_of_stems(
        (stand, cdata),
        max_proportion=float(max_prop),
        stems_after=float(stems_after),
        labels=["first_thinning"],
        sim_time=sim_time,
    )

    return (stand, cdata)

# ----------------------------------------------------------------------
# Events: Basal-area thinning from below – conifer priority family
# R references:
#   - event_ba_thinning_from_below_regular_coniferPriority.txt
#   - event_ba_thinning_from_below_regular_coniferPriority2.txt
#   - event_ba_thinning_from_below_regular_coniferPriority3.txt
#   - event_ba_thinning_from_below_regular_coniferBirchPriority.txt
# ----------------------------------------------------------------------

def _target_relative_g(stand: ForestStand, trees: ReferenceTrees, ba_after: Optional[float] = None) -> float:
    """Compute relative target on G as in R: (G_upper - ba_after [+2?]) / G. The +2 is encoded in the per-variant builders."""
    metrics = compute_stand_metrics(trees)
    G = metrics["G"]
    if ba_after is None:
        ba_after = basal_area_instruction_lower_limit(stand)
    G_upper = basal_area_instructions_upper_limit(stand)
    if G <= 0:
        return 0.0
    return max(0.0, min(1.0, (G_upper - ba_after) / G))

def build_selection_conifer_priority(stand: ForestStand, *, cap_other_share: float | None = None) -> dict[str, Any]:
    """
    Build selection sets used by the conifer priority variants.
    - Set 1: non-preferred group (other species), from-below, capped by a relative or absolute target depending on variant
    - Set 2: preferred conifer group, from-below, “take the rest” (relative=1)
    """
    s1, s2 = _sfunctions_conifer_priority(stand)

    return {
        "Target": {"type": "relative", "var": "g", "amount": None},  # filled by caller
        "sets": [
            {
                "sfunction": s1,
                "order_var": "d",           # diameter-ordered
                "target_var": "g",
                "target_type": "relative" if cap_other_share is not None else "relative",
                "target_amount": cap_other_share if cap_other_share is not None else 0.7,  # default cap (if used)
                "profile_x": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float),
                "profile_y": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.25, 0.1, 0.05, 0.05, 0.05], dtype=float),
                "profile_xmode": "relative",
            },
            {
                "sfunction": s2,
                "order_var": "d",
                "target_var": "g",
                "target_type": "relative",
                "target_amount": 1.0,
                "profile_x": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float),
                "profile_y": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.25, 0.1, 0.05, 0.05, 0.05], dtype=float),
                "profile_xmode": "relative",
            },
        ],
    }

def event_ba_thinning_from_below_regular_coniferPriority(
    op: OpTuple[ForestStand],
    *,
    sim_time: Optional[int] = None
) -> OpTuple[ForestStand]:
    """
    R logic: remove up to (G_upper - ba_after + 2)/G of G, from-below;
    first take non-preferred species (capped), then preferred conifers.
    """
    stand, cdata = op
    trees: ReferenceTrees = getattr(stand, "reference_trees", None)
    if trees is None:
        raise MetsiException("reference_trees (SoA) required")

    # R adds “+2” m2/ha on top of (upper-lower); preserve that here
    ba_after = basal_area_instruction_lower_limit(stand)
    metrics = compute_stand_metrics(trees)
    G = metrics["G"]
    if G <= 0:
        return (stand, cdata)

    rel_amount = max(0.0, min(1.0, (basal_area_instructions_upper_limit(stand) - ba_after + 2.0) / G))

    tree_selection = build_selection_conifer_priority(stand)
    tree_selection["Target"]["amount"] = rel_amount

    return ftrt_thin_basal_area(
        (stand, cdata),
        tree_selection=tree_selection,
        labels=["thinning_ba_coniferPriority"],
        sim_time=sim_time,
    )

def _other_species_mask(stand: ForestStand, trees: ReferenceTrees):
    s1, s2 = _sfunctions_conifer_priority(stand)
    return s1(stand, trees)

def event_ba_thinning_from_below_regular_coniferPriority2(op: OpTuple[ForestStand], *, sim_time: Optional[int] = None):
    stand, cdata = op
    trees: ReferenceTrees = getattr(stand, "reference_trees", None)
    if trees is None:
        raise MetsiException("reference_trees (SoA) required")
    metrics = compute_stand_metrics(trees)
    G = metrics["G"]
    if G <= 0:
        return (stand, cdata)

    ba_after = basal_area_instruction_lower_limit(stand)
    rel_amount = max(0.0, min(1.0, (basal_area_instructions_upper_limit(stand) - ba_after + 2.0) / G))

    sd = SelectionData(trees)
    mask_other = _other_species_mask(stand, trees)
    G_other = float(np.nansum(sd.g[mask_other] * trees.stems_per_ha[mask_other]))
    other_cap = 0.0
    if G_other > 0:
        # We want remaining 2 m2/ha, i.e. remove at most (G_other - 2) / G_other
        other_cap = max(0.0, min(1.0, (G_other - 2.0) / G_other))

    tree_selection = build_selection_conifer_priority(stand, cap_other_share=other_cap)
    tree_selection["Target"]["amount"] = rel_amount

    return ftrt_thin_basal_area((stand, cdata), tree_selection=tree_selection,
                                labels=["thinning_ba_coniferPriority2"], sim_time=sim_time)

def event_ba_thinning_from_below_regular_coniferPriority3(op: OpTuple[ForestStand], *, sim_time: Optional[int] = None):
    stand, cdata = op
    trees: ReferenceTrees = getattr(stand, "reference_trees", None)
    if trees is None:
        raise MetsiException("reference_trees (SoA) required")
    metrics = compute_stand_metrics(trees)
    G = metrics["G"]
    if G <= 0:
        return (stand, cdata)

    ba_after = basal_area_instruction_lower_limit(stand)
    rel_amount = max(0.0, min(1.0, (basal_area_instructions_upper_limit(stand) - ba_after + 2.0) / G))

    # Variant 3: keep an explicit other-species cap, e.g. 0.7 unless domain gives a different rule
    tree_selection = build_selection_conifer_priority(stand, cap_other_share=0.7)
    tree_selection["Target"]["amount"] = rel_amount

    return ftrt_thin_basal_area((stand, cdata), tree_selection=tree_selection,
                                labels=["thinning_ba_coniferPriority3"], sim_time=sim_time)

def event_ba_thinning_from_below_regular_coniferBirchPriority(op: OpTuple[ForestStand], *, sim_time: Optional[int] = None):
    """
    Flip priority toward birches (3/4) before conifers. Reuse the same shapes.
    """
    stand, cdata = op
    trees: ReferenceTrees = getattr(stand, "reference_trees", None)
    if trees is None:
        raise MetsiException("reference_trees (SoA) required")
    metrics = compute_stand_metrics(trees)
    G = metrics["G"]
    if G <= 0:
        return (stand, cdata)

    ba_after = basal_area_instruction_lower_limit(stand)
    rel_amount = max(0.0, min(1.0, (basal_area_instructions_upper_limit(stand) - ba_after + 2.0) / G))

    def s_birch(ctx, data): return (data["species"] == 3) | (data["species"] == 4)
    def s_other(ctx, data): return ~s_birch(ctx, data)

    prof_x = np.array([0,1,2,3,4,5,6,7,8,9,10], dtype=float)
    prof_y = np.array([0.5,0.5,0.5,0.5,0.5,0.4,0.25,0.1,0.05,0.05,0.05], dtype=float)
    tree_selection = {
        "Target": {"type":"relative","var":"g","amount": rel_amount},
        "sets": [
            {"sfunction": s_other, "order_var":"d","target_var":"g","target_type":"relative","target_amount":0.7,
             "profile_x": prof_x, "profile_y": prof_y, "profile_xmode":"relative"},
            {"sfunction": s_birch, "order_var":"d","target_var":"g","target_type":"relative","target_amount":1.0,
             "profile_x": prof_x, "profile_y": prof_y, "profile_xmode":"relative"},
        ],
    }

    return ftrt_thin_basal_area((stand, cdata), tree_selection=tree_selection,
                                labels=["thinning_ba_birchPriority"], sim_time=sim_time)