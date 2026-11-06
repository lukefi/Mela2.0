from dataclasses import dataclass
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


@dataclass(slots=True)
class RegenerationDecl:
    method: Origin
    species: TreeSpecies
    stems_per_ha: float
    height: float
    biological_age: float
    ntrees: int = 10
    breast_height_diameter: float | None = None
    breast_height_age: float | None = None
    delay: int = 0


def regeneration_fn(stand: ForestStand,
                    *,
                    regeneration_decls: list[RegenerationDecl] | None = None,
                    survival_percent_motti: float = 100.0) -> OpTuple[ForestStand]:
    # ----- obligatory params ----

    assert regeneration_decls is not None
    assert len(regeneration_decls) > 0

    # TODO: multiple separate types?
    regen_type = _resolve_regeneration_type(regeneration_decls[0].method)

    if stand.motti_state is not None:
        if len(regeneration_decls) == 2:
            # Mixed regeneration
            _regeneration_via_motti(
                stand,
                method=_resolve_method_from_origin(regeneration_decls[0].method),
                method_2=_resolve_method_from_origin(regeneration_decls[1].method),
                species=convert_species(regeneration_decls[0].species),
                species_2=convert_species(regeneration_decls[1].species),
                stems_per_ha=regeneration_decls[0].stems_per_ha,
                delay=regeneration_decls[0].delay,
                survival_percent=survival_percent_motti,
                species_ratio=(regeneration_decls[0].stems_per_ha / (regeneration_decls[0].stems_per_ha +
                                                                     regeneration_decls[1].stems_per_ha)))

        elif len(regeneration_decls) == 1:
            _regeneration_via_motti(
                stand,
                method=_resolve_method_from_origin(regeneration_decls[0].method),
                species=convert_species(regeneration_decls[0].species),
                stems_per_ha=regeneration_decls[0].stems_per_ha,
                delay=regeneration_decls[0].delay,
                survival_percent=survival_percent_motti)

        else:
            raise MetsiException("Motti regeneration only supported for one or two species")

    else:
        # Mixed regeneration
        for regen in regeneration_decls:
            for _ in range(regen.ntrees):
                identifier, tree_number = new_reference_tree_identity(stand)
                delayed_age = regen.biological_age - regen.delay
                stand.reference_trees.create({
                    "identifier": identifier,
                    "tree_number": tree_number,
                    "species": regen.species,
                    "origin": regen.method,
                    "stems_per_ha": regen.stems_per_ha / regen.ntrees,
                    "height": regen.height,
                    "biological_age": delayed_age if delayed_age > 0 else 0,
                    "breast_height_diameter": regen.breast_height_diameter,
                    "breast_height_age": regen.breast_height_age,
                    "management_category": TreeManagementCategory.NO_RESTRICTION})

    if regen_type == RegenerationType.ARTIFICIAL:
        # TODO: multiple separate delays?
        stand.artificial_regeneration_year = stand.year + regeneration_decls[0].delay

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
