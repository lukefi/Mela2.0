from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.conversion.internal2motti import convert_species, convert_soil_preparation_type
from lukefi.metsi.data.enums.internal import (
    Origin, RegenerationType, TreeSpecies, TreeManagementCategory
)
from lukefi.metsi.data.enums.motti import MottiRegenerationMethod, MottiSpecies
from lukefi.metsi.domain.natural_processes.motti_util import sync_ut_to_reference_trees
from lukefi.metsi.domain.natural_processes.util import new_reference_tree_identity
from lukefi.metsi.domain.natural_processes.motti_util import (
    prune_reference_trees_not_in_motti,
)
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.core.collected_data import OpTuple
from lukefi.metsi.core.exceptions import MetsiException
from lukefi.metsi.core.treatment import Treatment


def _is_cleared_after_cutting(stand: ForestStand) -> bool:
    if (stand.cutting_year is None or
        stand.regeneration_area_cleaning_year is None):
        return False
    if stand.cutting_year <= stand.regeneration_area_cleaning_year:
        return True
    return False


def _resolve_regeneration_type_from_origin(origin: Origin) -> RegenerationType:
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


# TODO: take the lower into use after storey PR is merged as it contains the TreeManagementCategory.SEEDING_TREE value.
# - https://github.com/lukefi/Mela2.0/pull/151
# def _resolve_seeding_tree_spe(rt: ReferenceTrees) -> MottiSpecies:
#     """ In-place resolution of Motti seeding tree value """
#     seeding_species = rt.species[rt.management_category == TreeManagementCategory.SEEDING_TREE]
#     all_possible_species, idx = np.unique(seeding_species, return_inverse=True)
#     result = all_possible_species[0]
#     if all_possible_species.size > 1:
#         # resolve which species has the larges basal area
#         sums = np.bincount(idx, weights=(rt.basal_area * rt.stems_per_ha))
#         result = all_possible_species[sums.argmax()]
#     return convert_species(result.item())


def _regeneration_via_motti(stand: ForestStand,
                            *,
                            origin: Origin,
                            species: TreeSpecies,
                            stems_per_ha: float,
                            motti_delay: int,
                            survival_percent: float = 100.0
                            ) -> None:
    assert stand.motti_state
    ms = stand.motti_state

    seed_tree_species = MottiSpecies.UNKNOWN
    # TODO: look the comments in the out commented function definition
    # if origin == Origin.NATURAL:
    #   seed_tree_species = _resolve_seeding_tree_spe(stand.reference_trees)

    motti_regeneration_params = [
        float(_resolve_method_from_origin(origin)),
        survival_percent,
        float(convert_species(species)),
        stems_per_ha,
        float(convert_soil_preparation_type(stand.soil_preparation_type)),
        float(_is_cleared_after_cutting(stand)),
        float(seed_tree_species),
    ]

    ms.ntrees = Motti4DLL.regenerate_with_state(
        ms.yy,
        ms.yp,
        int(ms.ntrees),
        ms.buffers,
        method_vec=motti_regeneration_params,
        motti_delay=motti_delay,
    )

    sync_ut_to_reference_trees(stand)
    prune_reference_trees_not_in_motti(stand) # Lopuksi vois tarkastella, että onko prunetuksella vaikutusta.


def regeneration_fn(input_: ForestStand,
                    /,
                    origin: Origin = Origin.UNSET,
                    species: TreeSpecies = TreeSpecies.UNSET,
                    stems_per_ha: float | None= None,
                    height: float | None = None,
                    biological_age: float | None = None,
                    breast_height_diameter: float | None = None,
                    breast_height_age: float | None = None,
                    ntrees: int = 10,
                    motti_delay: int = 0,
                    survival_percent_motti: float = 100.0
                    ) -> OpTuple[ForestStand]:
    """
    Regeneration treatment adds reference trees to a stand based on origin type
    - Parameters:
        origin:                         # e.g. 1 (natural), 2 (seeded) or 3 (planted)
        species:                        # tree species code
        stems_per_ha:                   # total stems/ha to distribute to created trees
        height:                         # initial height (m)
        biological_age:                 # biological age (years)
        breast_height_diameter:         # diameter (dm)
        breast_height_age:              # age at breat height (years)
        ntrees:                         # number of reference trees to create
    - If Motti defined as transition, delegates sapling regeneration to Motti4Regenerate with additional params:
        motti_delay:                    # delay in years before regeneration is realized for Motti saplings
        survival_percent_motti:         # value from 0.0 to 100.0
        soil_preparation_type_motti:    #  value from 0 to 6
        clearing_motti: bool            # Done or not done
    
    """
    stand = input_


    # ----- obligatory params ----

    if origin is None or origin == Origin.UNSET:
        raise MetsiException("Origin missing or not set")
    if species is None or species == TreeSpecies.UNSET:
        raise MetsiException("Species is missing or not set")
    if stems_per_ha is None:
        raise MetsiException("stems_per_ha is missing")
    if height is None:
        raise MetsiException("Height is missing")
    if biological_age is None:
        raise MetsiException("Biological age is missing")

    # ---- optional  params ----

    if height <= 0:
        raise MetsiException("Regeneration: Height can not be negative or zero")
    if not ntrees or ntrees <= 0:
        raise MetsiException("Parameter 'ntrees' must be positive")
    if stems_per_ha <= 0:
        raise MetsiException("Parameter 'stems_per_ha' must be > 0")

    regen_type = _resolve_regeneration_type_from_origin(origin)

    if stand.motti_state is not None:
        _regeneration_via_motti(
            stand,
            origin=origin,
            species=species,
            stems_per_ha=stems_per_ha,
            motti_delay=motti_delay,
            survival_percent=survival_percent_motti)
    else:
        per_tree_stems = stems_per_ha / float(ntrees)
        for _ in range(ntrees):
            identifier, tree_number = new_reference_tree_identity(stand)
            stand.reference_trees.create({
                "identifier": identifier,
                "tree_number": tree_number,
                "species": species,
                "origin": origin,
                "stems_per_ha": per_tree_stems,
                "height": height,
                "biological_age": biological_age,
                "breast_height_diameter": None if breast_height_diameter is None else float(breast_height_diameter),
                "breast_height_age": None if breast_height_age is None else float(breast_height_age),
                "management_category": TreeManagementCategory.NO_RESTRICTION
            })

    if regen_type == RegenerationType.ARTIFICIAL:
        stand.artificial_regeneration_year = stand.year

    return stand, []


regeneration = Treatment(regeneration_fn, "regeneration")
