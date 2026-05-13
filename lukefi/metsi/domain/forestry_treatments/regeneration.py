from typing import cast
from lukefi.metsi.data.conversion.internal2motti import convert_species
from lukefi.metsi.data.enums.internal import Origin, RegenerationType, TreeSpecies
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.sim.treatment import Treatment
from lukefi.metsi.domain.natural_processes.util import new_reference_tree_identity
from lukefi.metsi.domain.natural_processes.grow_motti import (
    sync_ut_to_reference_trees,
    prune_reference_trees_not_in_motti,
)
from lukefi.metsi.data.enums.motti import MottiRegenerationMethod


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
    prune_reference_trees_not_in_motti(stand)


def regeneration_fn(input_: ForestStand,
                    /,
                    origin: Origin | None = None,
                    species: TreeSpecies | None = None,
                    stems_per_ha: float | None = None,
                    height: float | None = None,
                    biological_age: float | None = None,
                    regen_type: RegenerationType | None = None,
                    method: MottiRegenerationMethod | None = None,
                    breast_height_diameter: float | None = None,
                    breast_height_age: float | None = None,
                    ntrees: int = 10,
                    istep: int = 0,
                    survival_percent: float = 100.0,
                    soil_preparation_type: int = 0,
                    clearing: int = 0,
                    seed_tree_species: TreeSpecies = TreeSpecies.UNKNOWN,
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
    if regen_type is None:
        raise MetsiException("regen_type is missing")

    # ---- optional ----

    if height <= 0:
        raise MetsiException("Regeneration: Height can not be negative or zero")
    if not ntrees or ntrees <= 0:
        raise MetsiException("Parameter 'ntrees' must be positive")
    if stems_per_ha <= 0:
        raise MetsiException("Parameter 'stems_per_ha' must be > 0")

    if regen_type == RegenerationType.ARTIFICIAL:
        stand.artificial_regeneration_year = stand.year

    if stand.motti_state is not None:
        if method is None:
            raise MetsiException("Regeneration method missing")

        _regeneration_via_motti(
            stand,
            method=method,
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
        })

    return stand, []


regeneration = Treatment(regeneration_fn, "regeneration")
