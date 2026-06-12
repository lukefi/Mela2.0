import numpy as np
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.data.enums.internal import Storey, TreeSpecies, SiteType
from lukefi.metsi.data.util.select_units import SelectionSet, SelectionTarget
from lukefi.metsi.sim.treatment import PredeterminedTreatment
from lukefi.metsi.forestry.harvest.cutting import cutting


def _osc_tree_set1_fn(stand: ForestStand, trees) -> np.ndarray:
    treesinset = (trees.storey == Storey.OVER) & \
        (trees.managementcategory <= 1)
    return treesinset


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

overStoreyCutting = PredeterminedTreatment(
    name="fc_over_storey_cutting",
    treatment_fn=cutting,
    static_parameters={
        "cutting_season": "random",
        "cutting_method": "cost_minimized",
        "cutting_skills": "professional",
        "tree_selection": _osc_tree_selection
    },
)


def _ft_remaining_stems(stand: ForestStand) -> float:
    '''
    Roughly based on https://metsanhoidonsuositukset.fi/fi/toimenpiteet/ensiharvennus/toteutus#section-387
    '''
    if stand.ds_main_tree_species == TreeSpecies.PINE:
        if stand.site_type_category in (SiteType.VERY_RICH_SITE,
                                        SiteType.RICH_SITE,
                                        SiteType.DAMP_SITE,
                                        SiteType.SUB_DRY_SITE):
            return 1100
        return 1000
    if stand.ds_main_tree_species == TreeSpecies.SPRUCE:
        if stand.region is not None and 17 <= stand.region <= 19:
            return 1200
        return 1000
    if stand.ds_main_tree_species == TreeSpecies.SILVER_BIRCH:
        return 800
    if stand.ds_main_tree_species == TreeSpecies.DOWNY_BIRCH:
        return 1200


def _ft_tree_base_set_fn(stand: ForestStand) -> np.ndarray:
    return np.logical_and(stand.reference_trees.breast_height_diameter >= 5,
                          stand.reference_trees.management_category <= 1)


def _ft_nstems_not_included(stand: ForestStand) -> float:
    trees_not_included = np.where(np.logical_not(_ft_tree_base_set_fn(stand)))
    return np.sum(stand.reference_trees.stems_per_ha[trees_not_included])


def _ft_tree_set1_fn(stand: ForestStand, trees) -> np.ndarray:
    trees_in_set = np.logical_and(trees.species != stand.ds_main_tree_species,
                                  _ft_tree_base_set_fn(stand))
    return trees_in_set


def _ft_tree_selection_fn(stand: ForestStand):
    return {
        "target": SelectionTarget(
            type_="absolute_remain",
            var="stems_per_ha",
            amount=ft_remaining_stems(stand) + _ft_nstems_not_included(stand),
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
                target_amount=ft_remaining_stems(stand),
                profile_x=[0, 1],
                profile_y=[1.0, 0.0],
                profile_xmode="relative"
            ),
        ],
    }


firstThinning = PredeterminedTreatment(
    name="fc_first_thinning",
    treatment_fn=cutting,
    static_parameters={
        "cutting_season": "random",
        "cutting_method": "cost_minimized",
        "cutting_skills": "professional"
    },
    dynamic_parameters={
        "tree_selection": _ft_tree_selection_fn
    }
)

# Dummy, will be replaced with real substance parameter function


def BA_AFTER_THINNING_BELOW(stand: ForestStand, trees) -> float:
    _site = stand.site_type_category
    _hdom = stand.ds_dominant_height
    _spe = stand.ds_main_tree_species

    if _site in (SiteType.VERY_RICH_SITE, SiteType.RICH_SITE, SiteType.DAMP_SITE):
        if _hdom <= 12:
            if _spe == TreeSpecies.PINE:
                return 15.3
            if _spe == TreeSpecies.SPRUCE:
                return 15.3
            if _spe == TreeSpecies.SILVER_BIRCH:
                return 8.5
            else:
                return 10.4
        else:
            if _spe == TreeSpecies.PINE:
                return 19.0
            if _spe == TreeSpecies.SPRUCE:
                return 20.0
            if _spe == TreeSpecies.SILVER_BIRCH:
                return 14.0
            else:
                return 13.4
    else:
        if _hdom <= 12:
            if _spe == TreeSpecies.PINE:
                return 14.0
            if _spe == TreeSpecies.SPRUCE:
                return 12.0
            else:
                return 10.4
        else:
            if _spe == TreeSpecies.PINE:
                return 16.0
            if _spe == TreeSpecies.SPRUCE:
                return 17.0
            else:
                return 13.4


def _thin_tree_base_set_fn(stand: ForestStand, trees) -> np.ndarray:
    return np.logical_and(trees.breast_height_diameter >= 6,
                          trees.management_category <= 1)


def _thin_ba_not_included(stand: ForestStand) -> float:
    trees_not_included = np.where(np.logical_not(_thin_tree_base_set_fn(stand)))
    return np.sum(stand.reference_trees.basal_area[trees_not_included])


def _thin_tree_selection_fn(stand: ForestStand):
    return {
        "target": SelectionTarget(
            type_="absolute_remain",
            var="basal_area",
            amount=BA_AFTER_THINNING_BELOW(stand) + _thin_ba_not_included(stand),
        ),
        "sets": [
            SelectionSet[ForestStand, ReferenceTrees](
                sfunction=_thin_tree_base_set_fn,
                order_var="breast_height_diameter",
                target_var="basal_area",
                target_type="absolute_remain",
                target_amount=BA_AFTER_THINNING_BELOW(stand) + _thin_ba_not_included(stand),
                profile_x=[0, 1],
                profile_y=[1.0, 0.0],
                profile_xmode="relative"
            ),
        ],
    }


thinning = PredeterminedTreatment(
    name="fc_thinning",
    treatment_fn=cutting,
    static_parameters={
        "cutting_season": "random",
        "cutting_method": "cost_minimized",
        "cutting_skills": "professional"
    },
    dynamic_parameters={
        "tree_selection": _thin_tree_selection_fn
    }
)


def _cc_tree_set1_fn(stand: ForestStand, trees) -> np.ndarray:
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

clearCutting = PredeterminedTreatment(
    name="fc_clear_cutting",
    treatment_fn=cutting,
    static_parameters={
        "cutting_season": "random",
        "cutting_method": "cost_minimized",
        "cutting_skills": "professional",
        "tree_selection": _cc_tree_selection
    },
)
