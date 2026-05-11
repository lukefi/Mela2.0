from typing import cast
from lukefi.metsi.data.conversion.internal2motti import convert_species
from lukefi.metsi.data.enums.internal import Origin, RegenerationType, TreeSpecies
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.forestry.treatment_utils import req
from lukefi.metsi.sim.treatment import Treatment
from lukefi.metsi.domain.natural_processes.util import new_reference_tree_identity
from lukefi.metsi.domain.natural_processes.grow_motti_dll import (
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


def regeneration_fn(input_: ForestStand, /, **operation_parameters) -> OpTuple[ForestStand]:
    """
    Regeneration treatment: add *reference trees*.
    - No cdata collection by design.
    - Parameters (all via **operation_parameters):
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

    origin: Origin = req(operation_parameters, "origin")
    species: TreeSpecies = req(operation_parameters, "species")
    stems_per_ha = float(req(operation_parameters, "stems_per_ha"))
    height = float(req(operation_parameters, "height"))
    biological_age = float(req(operation_parameters, "biological_age"))
    regen_type: RegenerationType = req(operation_parameters, "type")

    # ---- optional ----
    method = cast(MottiRegenerationMethod | None, operation_parameters.get("method", None))
    breast_height_diameter = operation_parameters.get("breast_height_diameter", None)
    breast_height_age = operation_parameters.get("breast_height_age", None)
    ntrees = operation_parameters.get("ntrees", 10)

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
            step=int(operation_parameters.get("istep", 0)),
            survival_percent=float(operation_parameters.get("survival_percent", 100.0)),
            soil_preparation_type=int(operation_parameters.get("soil_preparation_type", 0)),
            clearing=int(operation_parameters.get("clearing", 0)),
            seed_tree_species=operation_parameters.get("seed_tree_species", TreeSpecies.UNKNOWN),
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
