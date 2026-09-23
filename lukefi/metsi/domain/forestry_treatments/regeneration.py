from lukefi.metsi.data.conversion.internal2motti import convert_species
from lukefi.metsi.data.enums.internal import Origin, RegenerationType, TreeSpecies, TreeManagementCategory
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.enums.motti import MottiRegenerationMethod
from lukefi.metsi.domain.natural_processes.motti_util import sync_ut_to_reference_trees
from lukefi.metsi.domain.natural_processes.util import new_reference_tree_identity
from lukefi.metsi.domain.natural_processes.motti_util import (
    prune_reference_trees_not_in_motti,
)
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.core.collected_data import OpTuple
from lukefi.metsi.core.exceptions import MetsiException
from lukefi.metsi.core.treatment import Treatment

def _resolve_regeneration_type_from_origin(origin: Origin) -> RegenerationType:
    """ In-place resolution from origin to Motti regeneration type coding """
    _natural_origins = (Origin.UNSET, Origin.NATURAL)
    _artificial_origins = (Origin.SEEDED, Origin.PLANTED)

    # NOTE: Should Motti have its own RegenerationType enum although it would be the same as internal?
    if origin in _natural_origins:
        return RegenerationType.NATURAL
    if origin in _artificial_origins:
        return RegenerationType.ARTIFICIAL

    raise MetsiException(f"Unable to solve Motti regeneration type base on stand origin value {origin}")

def _resolve_method_from_origin(origin: Origin) -> MottiRegenerationMethod:
    """ In-place resolution from origin to Motti regeneration method coding """
    # NOTE: Should this be in internal2motti?
    method_map = {
        # NOTE: Should we also check unset to natural? I think it is mainly done earlier?
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


def regeneration_fn(input_: ForestStand,
                    /,
                    origin: Origin | None = None,
                    species: TreeSpecies | None = None,
                    stems_per_ha: float | None = None,
                    height: float | None = None,
                    biological_age: float | None = None,
                    breast_height_diameter: float | None = None,
                    breast_height_age: float | None = None,
                    ntrees: int = 10,
                    survival_percent: float = 100.0,
                    istep: int = 0, # Jos realisoituu out parametriksi, niin pois
                    soil_preparation_type: int = 0, # Tämä pois ja katsotaan standista suoraan. (Kunhan ensin lisätään FDM)
                    clearing: int = 0,
                    seed_tree_species: TreeSpecies = TreeSpecies.UNKNOWN # Siemenpuutieto maskilla rt:stä ja päättely max(pl. ppa for all rt)
                    ) -> OpTuple[ForestStand]:
    """
    Regeneration treatment: add *reference trees*.
    - No cdata collection by design.
    - Parameters:
        origin: int                 # e.g. 2 (planted)
        method: Optional[int]       # accepted, unused
        species: int                # tree species code
        stems_per_ha: float         # total stems/ha to distribute to created trees
        height: float               # initial height (m)
        biological_age: float       # biological age (years)
        breast_height_diameter: Optional[float] = None
        breast_height_age: Optional[float] = None
        ntrees: Optional[int] = 10  # number of reference trees to create
        labels: Optional[list[str]] = None  # accepted, unused
        type: str                   # "artificial" | "natural"

    - Motti path: if stand.motti_state exists, delegate sapling regeneration to Motti4Regenerate
    """
    stand = input_

    if origin is None:
        raise MetsiException("Origin missing")
    if species is None:
        raise MetsiException("Species is missing")
    if stems_per_ha is None:
        raise MetsiException("stems_per_ha is missing")
    if height is None:
        raise MetsiException("Height is missing")
    if biological_age is None:
        raise MetsiException("Biological age is missing")

    # ---- optional ----

    if height <= 0:
        raise MetsiException("Regeneration: Height can not be negative or zero")
    if not ntrees or ntrees <= 0:
        raise MetsiException("Parameter 'ntrees' must be positive")
    if stems_per_ha <= 0:
        raise MetsiException("Parameter 'stems_per_ha' must be > 0")

    regen_type = _resolve_regeneration_type_from_origin(origin)

    # NOTE: Should this be after the actual treatment call?
    if regen_type == RegenerationType.ARTIFICIAL:
        stand.artificial_regeneration_year = stand.year

        _regeneration_via_motti(
            stand,
            method=_resolve_method_from_origin(origin),
            species=species,
            stems_per_ha=stems_per_ha,
            step=istep,
            survival_percent=survival_percent,
            soil_preparation_type=soil_preparation_type,
            clearing=clearing,
            seed_tree_species=seed_tree_species,
        )
        return stand, []

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

    return stand, []


def _regeneration_via_motti(stand: ForestStand,
                            *,
                            method: MottiRegenerationMethod,
                            species: TreeSpecies,
                            stems_per_ha: float,
                            step: int,
                            survival_percent: float = 100.0,
                            soil_preparation_type: int = 0,
                            clearing: int = 0,
                            seed_tree_species: TreeSpecies = TreeSpecies.UNKNOWN,
                            ) -> None:
    ms = stand.motti_state
    if ms is None or ms.buffers is None:
        raise MetsiException("Motti regeneration requested but stand has no initialized motti_state")

    cultivated_species = convert_species(species)
    seed_species = convert_species(seed_tree_species)

    method_vec = [
        float(method),
        survival_percent,
        float(cultivated_species),
        stems_per_ha,
        soil_preparation_type,
        clearing,
        float(seed_species),
    ]

    ms.ntrees = Motti4DLL.regenerate_with_state(
        ms.yy,
        ms.yp,
        int(ms.ntrees),
        ms.buffers,
        method=method_vec,
        step=int(step),
    )

    sync_ut_to_reference_trees(stand)
    prune_reference_trees_not_in_motti(stand) # Lopuksi vois tarkastella, että onko prunetuksella vaikutusta.


regeneration = Treatment(regeneration_fn, "regeneration")
