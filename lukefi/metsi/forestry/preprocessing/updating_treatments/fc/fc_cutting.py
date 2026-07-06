import numpy as np
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.data.enums.internal import Storey, TreeSpecies, SiteType
from lukefi.metsi.data.util.select_units import SelectionSet, SelectionTarget
from lukefi.metsi.domain.collected_data import RemovedTrees
from lukefi.metsi.sim.treatment import PredeterminedTreatment
from lukefi.metsi.forestry.harvest.cutting import cutting_fn
from lukefi.metsi.sim.utils import MetsiException


def _osc_tree_set1_fn(_: ForestStand, trees) -> np.ndarray:
    return (trees.storey == Storey.OVER) & (trees.managementcategory <= 1)


_osc_tree_selection = {
    "target": SelectionTarget(
        type_="relative",
        var="basal_area",
        amount=1
    ),
    "sets": [
        SelectionSet[ForestStand, ReferenceTrees](
            sfunction=_osc_tree_set1_fn,
            order_var="breast_height_diameter",
            target_var="stems_per_ha",
            target_type="relative",
            target_amount=1,
            profile_x=[0, 1],
            profile_y=[0.5, 0.5],
            profile_xmode="relative"
        ),
    ],
}

over_storey_cutting = PredeterminedTreatment(
    name="fc_over_storey_cutting",
    treatment_fn=cutting_fn,
    static_parameters={
        "cutting_season": "random",
        "cutting_method": "cost_minimized",
        "cutting_skills": "professional",
        "tree_selection": _osc_tree_selection
    },
    collected_data={RemovedTrees}
)


def _ft_remaining_stems(stand: ForestStand) -> float:
    '''
    Roughly based on https://metsanhoidonsuositukset.fi/fi/toimenpiteet/ensiharvennus/toteutus#section-387
    '''
    if stand.main_tree_species_dominant_storey == TreeSpecies.PINE:
        if stand.site_type_category in (SiteType.VERY_RICH_SITE,
                                        SiteType.RICH_SITE,
                                        SiteType.DAMP_SITE,
                                        SiteType.SUB_DRY_SITE):
            return 1100
        return 1000
    if stand.main_tree_species_dominant_storey == TreeSpecies.SPRUCE:
        if stand.region is not None and 17 <= stand.region <= 19:
            return 1200
        return 1000
    if stand.main_tree_species_dominant_storey == TreeSpecies.SILVER_BIRCH:
        return 800
    if stand.main_tree_species_dominant_storey == TreeSpecies.DOWNY_BIRCH:
        return 1200

    raise MetsiException(f"Unsupported ds_main_tree_species {stand.main_tree_species_dominant_storey}")


def _ft_tree_base_set_fn(stand: ForestStand, _: ReferenceTrees | None = None) -> np.ndarray:
    return np.logical_and(stand.reference_trees.breast_height_diameter >= 5,
                          stand.reference_trees.management_category <= 1)


def _ft_nstems_not_included(stand: ForestStand) -> float:
    trees_not_included = np.where(np.logical_not(_ft_tree_base_set_fn(stand)))
    return np.sum(stand.reference_trees.stems_per_ha[trees_not_included])


def _ft_tree_set1_fn(stand: ForestStand, trees: ReferenceTrees) -> np.ndarray:
    trees_in_set = np.logical_and(trees.species != stand.main_tree_species_dominant_storey,
                                  _ft_tree_base_set_fn(stand))
    return trees_in_set


def _ft_tree_selection_fn(stand: ForestStand):
    return {
        "target": SelectionTarget(
            type_="absolute_remain",
            var="stems_per_ha",
            amount=_ft_remaining_stems(stand) + _ft_nstems_not_included(stand),
        ),
        "sets": [
            SelectionSet[ForestStand, ReferenceTrees](
                sfunction=_ft_tree_set1_fn,
                order_var="breast_height_diameter",
                target_var="stems_per_ha",
                target_type="relative",
                target_amount=1,
                profile_x=[0, 1],
                profile_y=[1.0, 0.0],
                profile_xmode="relative"
            ),
            SelectionSet[ForestStand, ReferenceTrees](
                sfunction=_ft_tree_base_set_fn,
                order_var="breast_height_diameter",
                target_var="stems_per_ha",
                target_type="absolute_remain",
                target_amount=_ft_remaining_stems(stand),
                profile_x=[0, 1],
                profile_y=[1.0, 0.0],
                profile_xmode="relative"
            ),
        ],
    }


first_thinning = PredeterminedTreatment(
    name="fc_first_thinning",
    treatment_fn=cutting_fn,
    static_parameters={
        "cutting_season": "random",
        "cutting_method": "cost_minimized",
        "cutting_skills": "professional"
    },
    dynamic_parameters={
        "tree_selection": _ft_tree_selection_fn
    },
    collected_data={RemovedTrees}
)

# Dummy, will be replaced with real substance parameter function


def _ba_after_thinning_below(stand: ForestStand, _: ReferenceTrees | None = None) -> float:
    _site = stand.site_type_category
    _hdom = stand.ds_dominant_height
    _spe = stand.main_tree_species_dominant_storey

    assert _hdom is not None

    if _site in (SiteType.VERY_RICH_SITE, SiteType.RICH_SITE, SiteType.DAMP_SITE):
        if _hdom <= 12:
            if _spe == TreeSpecies.PINE:
                return 15.3
            if _spe == TreeSpecies.SPRUCE:
                return 15.3
            if _spe == TreeSpecies.SILVER_BIRCH:
                return 8.5
            return 10.4
        if _spe == TreeSpecies.PINE:
            return 19.0
        if _spe == TreeSpecies.SPRUCE:
            return 20.0
        if _spe == TreeSpecies.SILVER_BIRCH:
            return 14.0
        return 13.4

    if _hdom <= 12:
        if _spe == TreeSpecies.PINE:
            return 14.0
        if _spe == TreeSpecies.SPRUCE:
            return 12.0
        return 10.4
    if _spe == TreeSpecies.PINE:
        return 16.0
    if _spe == TreeSpecies.SPRUCE:
        return 17.0
    return 13.4


def _thin_tree_base_set_fn(_: ForestStand, trees: ReferenceTrees) -> np.ndarray:
    return np.logical_and(trees.breast_height_diameter >= 6,
                          trees.management_category <= 1)


def _thin_ba_not_included(stand: ForestStand, trees: ReferenceTrees) -> float:
    trees_not_included = np.where(np.logical_not(_thin_tree_base_set_fn(stand, trees)))
    return np.sum(stand.reference_trees.basal_area[trees_not_included])


def _thin_tree_selection_fn(stand: ForestStand):
    return {
        "target": SelectionTarget(
            type_="absolute_remain",
            var="basal_area",
            amount=_ba_after_thinning_below(stand) + _thin_ba_not_included(stand, stand.reference_trees),
        ),
        "sets": [
            SelectionSet[ForestStand, ReferenceTrees](
                sfunction=_thin_tree_base_set_fn,
                order_var="breast_height_diameter",
                target_var="basal_area",
                target_type="absolute_remain",
                target_amount=_ba_after_thinning_below(stand) + _thin_ba_not_included(stand, stand.reference_trees),
                profile_x=[0, 1],
                profile_y=[1.0, 0.0],
                profile_xmode="relative"
            ),
        ],
    }


thinning = PredeterminedTreatment(
    name="fc_thinning",
    treatment_fn=cutting_fn,
    static_parameters={
        "cutting_season": "random",
        "cutting_method": "cost_minimized",
        "cutting_skills": "professional"
    },
    dynamic_parameters={
        "tree_selection": _thin_tree_selection_fn
    },
    collected_data={RemovedTrees}
)


def _cc_tree_set1_fn(_: ForestStand, trees) -> np.ndarray:
    trees_in_set = (trees.breast_height_diameter >= 6) & \
                   (trees.management_category <= 1)
    return trees_in_set


_cc_tree_selection = {
    "target": SelectionTarget(
        type_="relative",
        var="basal_area",
        amount=1,
    ),
    "sets": [
        SelectionSet[ForestStand, ReferenceTrees](
            sfunction=_cc_tree_set1_fn,
            order_var="breast_height_diameter",
            target_var="basal_area",
            target_type="relative",
            target_amount=1,
            profile_x=[0, 1],
            profile_y=[0.5, 0.5],
            profile_xmode="relative"
        ),
    ],
}

clearcutting = PredeterminedTreatment(
    name="fc_clear_cutting",
    treatment_fn=cutting_fn,
    static_parameters={
        "cutting_season": "random",
        "cutting_method": "cost_minimized",
        "cutting_skills": "professional",
        "tree_selection": _cc_tree_selection
    },
    collected_data={RemovedTrees}
)
