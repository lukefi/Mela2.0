from typing import overload

import numpy as np

from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.conversion.internal2motti import convert_species, convert_soil_preparation_type
from lukefi.metsi.data.enums.internal import (
    Origin, RegenerationType, Storey, TreeSpecies, TreeManagementCategory
)
from lukefi.metsi.data.enums.motti import MottiRegenerationMethod, MottiSpecies
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.domain.natural_processes.motti_util import sync_ut_to_reference_trees
from lukefi.metsi.domain.natural_processes.util import new_reference_tree_identity
from lukefi.metsi.domain.natural_processes.motti_util import (
    prune_reference_trees_not_in_motti,
)
from lukefi.metsi.forestry.aggregates_internal import calc_mean_height, calc_tree_basal_areas
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.core.collected_data import OpTuple
from lukefi.metsi.core.exceptions import MetsiException
from lukefi.metsi.core.treatment import Treatment

SPECIES_SUITABLE_FOR_MIXED_REGENERATION = {TreeSpecies.PINE, TreeSpecies.SPRUCE, TreeSpecies.SILVER_BIRCH}


def _is_cleared_after_cutting(stand: ForestStand) -> bool:
    if (stand.cutting_year is None or
            stand.regeneration_area_cleaning_year is None):
        return False
    if stand.cutting_year <= stand.regeneration_area_cleaning_year:
        return True
    return False


def _resolve_regeneration_type(origin: Origin) -> RegenerationType:
    """ In-place resolution from origin to Motti regeneration type coding """
    _natural_origins = (Origin.UNSET, Origin.NATURAL)
    _artificial_origins = (Origin.SEEDED, Origin.PLANTED)

    if origin in _natural_origins:
        return RegenerationType.NATURAL
    if origin in _artificial_origins:
        return RegenerationType.ARTIFICIAL

    raise MetsiException(f"Unable to solve Motti regeneration type base on stand origin value {origin}")


def _resolve_method_from_origin(origin: Origin) -> MottiRegenerationMethod:
    """ In-place resolution from origin to Motti regeneration method coding """
    method_map = {
        Origin.NATURAL: MottiRegenerationMethod.NATURAL,
        Origin.SEEDED: MottiRegenerationMethod.SOWING,
        Origin.PLANTED: MottiRegenerationMethod.PLANTING
    }
    try:
        result = method_map[origin]
    except KeyError as e:
        raise MetsiException(
            f"Unable to resolve Motti regeneration method coding based on stand origin value {origin}"
        ) from e
    return result


def _resolve_seeding_tree_spe(rt: ReferenceTrees) -> MottiSpecies:
    """ In-place resolution of Motti seeding tree value """
    seeding_species = rt.species[rt.management_category == TreeManagementCategory.SEEDING_TREE]
    all_possible_species, idx = np.unique(seeding_species, return_inverse=True)
    result = all_possible_species[0]
    if all_possible_species.size > 1:
        # resolve which species has the larges basal area
        sums = np.bincount(idx, weights=rt.basal_area * rt.stems_per_ha)
        result = all_possible_species[sums.argmax()]
    return convert_species(result.item())


@overload
def _regeneration_via_motti(stand: ForestStand,
                            *,
                            method: MottiRegenerationMethod,
                            survival_percent: float = 100.0,
                            species: MottiSpecies,
                            stems_per_ha: float,
                            delay: int):
    pass


@overload
def _regeneration_via_motti(stand: ForestStand,
                            *,
                            method: MottiRegenerationMethod,
                            method_2: MottiRegenerationMethod,
                            survival_percent: float = 100.0,
                            species: MottiSpecies,
                            species_2: MottiSpecies,
                            stems_per_ha: float,
                            species_ratio: float,
                            delay: int):
    pass


def _regeneration_via_motti(stand: ForestStand,
                            *,
                            method: MottiRegenerationMethod,
                            method_2: MottiRegenerationMethod | None = None,
                            survival_percent: float = 100.0,
                            species: MottiSpecies,
                            species_2: MottiSpecies | None = None,
                            stems_per_ha: float,
                            species_ratio: float | None = None,
                            delay: int) -> None:
    assert stand.motti_state
    ms = stand.motti_state

    seed_tree_species = MottiSpecies.UNKNOWN
    if method == MottiRegenerationMethod.NATURAL:
        seed_tree_species = _resolve_seeding_tree_spe(stand.reference_trees)

    if method_2 is not None:
        assert species_2 is not None
        assert species_ratio is not None
        method_f = float(method) * 10 + float(method_2)
        species_f = float(species) * 10 + float(species_2)
    else:
        method_f = float(method)
        species_f = float(species)
        species_ratio = 0

    motti_regeneration_params = [
        method_f,
        survival_percent,
        species_f,
        stems_per_ha,
        float(convert_soil_preparation_type(stand.soil_preparation_type)),
        float(_is_cleared_after_cutting(stand)),
        float(seed_tree_species),
        species_ratio
    ]

    ms.ntrees = Motti4DLL.regenerate_with_state(
        ms.yy,
        ms.yp,
        int(ms.ntrees),
        ms.buffers,
        method_vec=motti_regeneration_params,
        delay=delay,
    )

    sync_ut_to_reference_trees(stand)
    prune_reference_trees_not_in_motti(stand)  # Lopuksi vois tarkastella, että onko prunetuksella vaikutusta.


@overload
def regeneration_fn(stand: ForestStand,
                    *,
                    method: Origin = Origin.UNSET,
                    species: TreeSpecies = TreeSpecies.UNSET,
                    stems_per_ha: float | None = None,
                    height: float | None = None,
                    biological_age: float | None = None,
                    breast_height_diameter: float | None = None,
                    breast_height_age: float | None = None,
                    ntrees: int = 10,
                    delay: int = 0,
                    survival_percent_motti: float = 100.0) -> OpTuple[ForestStand]:
    """
    Regeneration treatment adds reference trees to a stand based on method
    - Parameters:
        method:                         # regeneration method e.g. 1 (natural), 2 (seeded) or 3 (planted)
        species:                        # tree species code
        stems_per_ha:                   # total stems/ha to distribute to created trees
        height:                         # initial height (m)
        biological_age:                 # biological age (years)
        breast_height_diameter:         # diameter (dm)
        breast_height_age:              # age at breat height (years)
        ntrees:                         # number of reference trees to create
        delay:                          # delay in years before regeneration is done
    - If Motti defined as transition, delegates sapling regeneration to Motti4Regenerate with additional params:
        survival_percent_motti:         # value from 0.0 to 100.0
        soil_preparation_type_motti:    # value from 0 to 6
        clearing_motti: bool            # Done or not done

    """


@overload
def regeneration_fn(stand: ForestStand,
                    *,
                    method: Origin = Origin.UNSET,
                    method_2: Origin = Origin.UNSET,
                    species: TreeSpecies = TreeSpecies.UNSET,
                    species_2: TreeSpecies = TreeSpecies.UNSET,
                    stems_per_ha: float | None = None,
                    stems_per_ha_2: float | None = None,
                    height: float | None = None,
                    height_2: float | None = None,
                    biological_age: float | None = None,
                    biological_age_2: float | None = None,
                    breast_height_diameter: float | None = None,
                    breast_height_diameter_2: float | None = None,
                    breast_height_age: float | None = None,
                    breast_height_age_2: float | None = None,
                    ntrees: int = 10,
                    ntrees_2: int = 0,
                    delay: int = 0,
                    delay_2: int = 0,
                    survival_percent_motti: float = 100.0) -> OpTuple[ForestStand]:
    """
    Mixed regeneration treatment adds reference trees to a stand based on method
    - Parameters:
        method:                         # regeneration method of first species e.g. 1 (natural), 2 (seeded) or 3
                                        # (planted)
        method_2:                       # regeneration method of second species e.g. 1 (natural), 2 (seeded) or 3
                                        # (planted)
        species:                        # first species
        species_2:                      # second species
        stems_per_ha:                   # total stems/ha to distribute to created trees of the first species
        stems_per_ha_2:                 # total stems/ha to distribute to created trees of the second species
        height:                         # initial height of the first species (m)
        height_2:                       # initial height of the second species (m)
        biological_age:                 # biological age of the first species (years)
        biological_age_2:               # biological age of the second species (years)
        breast_height_diameter:         # diameter of the first species (dm)
        breast_height_diameter_2:       # diameter of the second species (dm)
        breast_height_age:              # age at breat height of the first species (years)
        breast_height_age_2:            # age at breat height of the second species (years)
        ntrees:                         # number of reference trees of the first species to create
        ntrees_2:                       # number of reference trees of the second species to create
        delay:                          # delay in years before regeneration of the first species is done
        delay_2:                          # delay in years before regeneration of the second species is done
    - If Motti defined as transition, delegates sapling regeneration to Motti4Regenerate with additional params:
        survival_percent_motti:         # value from 0.0 to 100.0
        soil_preparation_type_motti:    # value from 0 to 6
        clearing_motti: bool            # Done or not done

    """


def regeneration_fn(stand: ForestStand,
                    *,
                    method: Origin = Origin.UNSET,
                    method_2: Origin = Origin.UNSET,
                    species: TreeSpecies = TreeSpecies.UNSET,
                    species_2: TreeSpecies = TreeSpecies.UNSET,
                    stems_per_ha: float | None = None,
                    stems_per_ha_2: float | None = None,
                    height: float | None = None,
                    height_2: float | None = None,
                    biological_age: float | None = None,
                    biological_age_2: float | None = None,
                    breast_height_diameter: float | None = None,
                    breast_height_diameter_2: float | None = None,
                    breast_height_age: float | None = None,
                    breast_height_age_2: float | None = None,
                    ntrees: int = 10,
                    ntrees_2: int = 0,
                    delay: int = 0,
                    delay_2: int = 0,
                    survival_percent_motti: float = 100.0) -> OpTuple[ForestStand]:
    # ----- obligatory params ----

    if method == Origin.UNSET:
        raise MetsiException("Origin missing or not set")
    if species == TreeSpecies.UNSET:
        raise MetsiException("Species is missing or not set")
    if stems_per_ha is None:
        raise MetsiException("stems_per_ha is missing")
    if height is None:
        raise MetsiException("Height is missing")
    if biological_age is None:
        raise MetsiException("Biological age is missing")

    if method_2 != Origin.UNSET:
        if Origin.NATURAL in {method, method_2}:
            raise MetsiException("method and method_2 must be SEEDED or PLANTED for mixed regeneration")
        if (species not in SPECIES_SUITABLE_FOR_MIXED_REGENERATION or
                species_2 not in SPECIES_SUITABLE_FOR_MIXED_REGENERATION):
            raise MetsiException("species and species_2 must be pine, spruce or silver birch for mixed regeneration")
        if stems_per_ha_2 is None:
            raise MetsiException("stems_per_ha_2 missing for mixed regeneration")

    # ---- optional  params ----

    if height <= 0:
        raise MetsiException("Regeneration: Height can not be negative or zero")
    if not ntrees or ntrees <= 0:
        raise MetsiException("Parameter 'ntrees' must be positive")
    if stems_per_ha <= 0:
        raise MetsiException("Parameter 'stems_per_ha' must be > 0")

    regen_type = _resolve_regeneration_type(method)

    if stand.motti_state is not None:
        if method_2 != Origin.UNSET and species_2 != TreeSpecies.UNSET and stems_per_ha_2 is not None:
            # Mixed regeneration
            _regeneration_via_motti(
                stand,
                method=_resolve_method_from_origin(method),
                method_2=_resolve_method_from_origin(method_2),
                species=convert_species(species),
                species_2=convert_species(species_2),
                stems_per_ha=stems_per_ha,
                delay=delay,
                survival_percent=survival_percent_motti,
                species_ratio=stems_per_ha / (stems_per_ha + stems_per_ha_2))
        else:
            _regeneration_via_motti(
                stand,
                method=_resolve_method_from_origin(method),
                species=convert_species(species),
                stems_per_ha=stems_per_ha,
                delay=delay,
                survival_percent=survival_percent_motti)

    elif method_2 == Origin.UNSET:
        per_tree_stems = stems_per_ha / float(ntrees)
        for _ in range(ntrees):
            identifier, tree_number = new_reference_tree_identity(stand)
            delayed_age = biological_age - delay
            stand.reference_trees.create({
                "identifier": identifier,
                "tree_number": tree_number,
                "species": species,
                "origin": method,
                "stems_per_ha": per_tree_stems,
                "height": height,
                "biological_age": delayed_age if delayed_age > 0 else 0,
                "breast_height_diameter": None if breast_height_diameter is None else float(breast_height_diameter),
                "breast_height_age": None if breast_height_age is None else float(breast_height_age),
                "management_category": TreeManagementCategory.NO_RESTRICTION
            })
    else:
        # Mixed regeneration
        assert species_2 != TreeSpecies.UNSET
        assert ntrees_2 > 0
        assert height_2 is not None
        assert biological_age_2 is not None
        assert stems_per_ha_2 is not None

        per_tree_stems_1 = stems_per_ha / ntrees
        per_tree_stems_2 = stems_per_ha_2 / ntrees_2

        for _ in range(ntrees):
            identifier, tree_number = new_reference_tree_identity(stand)
            delayed_age = biological_age - delay
            stand.reference_trees.create({
                "identifier": identifier,
                "tree_number": tree_number,
                "species": species,
                "origin": method,
                "stems_per_ha": per_tree_stems_1,
                "height": height,
                "biological_age": delayed_age if delayed_age > 0 else 0,
                "breast_height_diameter": None if breast_height_diameter is None else float(breast_height_diameter),
                "breast_height_age": None if breast_height_age is None else float(breast_height_age),
                "management_category": TreeManagementCategory.NO_RESTRICTION})

        for _ in range(ntrees_2):
            identifier, tree_number = new_reference_tree_identity(stand)
            delayed_age = biological_age_2 - delay_2
            stand.reference_trees.create({
                "identifier": identifier,
                "tree_number": tree_number,
                "species": species_2,
                "origin": method_2,
                "stems_per_ha": per_tree_stems_2,
                "height": height_2,
                "biological_age": delayed_age if delayed_age > 0 else 0,
                "breast_height_diameter": None if breast_height_diameter_2 is None else float(breast_height_diameter_2),
                "breast_height_age": None if breast_height_age_2 is None else float(breast_height_age_2),
                "management_category": TreeManagementCategory.NO_RESTRICTION})

    if regen_type == RegenerationType.ARTIFICIAL:
        stand.artificial_regeneration_year = stand.year + delay

    # Handle storeys ----------------------------------------------------------------------------------------------

    trees = stand.reference_trees
    if np.any(trees.storey == Storey.DOMINANT):
        # Pre-calculate basal area for all trees
        trees.basal_area = calc_tree_basal_areas(trees.breast_height_diameter)

        mean_heights = {storey: calc_mean_height(trees, trees.storey == storey)
                        for storey in (Storey.DOMINANT, Storey.UNSET)}
        diff = mean_heights[Storey.DOMINANT] - mean_heights[Storey.UNSET]
        if diff >= 5.0:
            # Merge old DOMINANT into OVER
            trees.storey[trees.storey == Storey.DOMINANT] = Storey.OVER

    # Promote new trees to DOMINANT storey
    trees.storey[trees.storey == Storey.UNSET] = Storey.DOMINANT

    return stand, []


regeneration = Treatment(regeneration_fn, "regeneration")
