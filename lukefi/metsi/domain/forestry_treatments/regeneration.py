from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.forestry.treatment_utils import req
from lukefi.metsi.sim.treatment import Treatment
from lukefi.metsi.domain.natural_processes.util import new_reference_tree_identity
from lukefi.metsi.domain.natural_processes.grow_motti_dll import (
    species_to_motti,
    sync_ut_to_reference_trees,
    prune_reference_trees_not_in_motti,
)


def _regeneration_via_motti(
    stand: ForestStand,
    *,
    method: int,
    species: int,
    stems_per_ha: float,
    step: int,
    survival_percent: float = 100.0,
    soil_preparation_type: int = 0,
    clearing: int = 0,
    seed_tree_species: int = 0,
) -> None:
    ms = stand.motti_state
    if ms is None or ms.buffers is None:
        raise MetsiException("Motti regeneration requested but stand has no initialized motti_state")

    cultivated_species = species_to_motti(species)
    seed_species = species_to_motti(seed_tree_species) if seed_tree_species else 0

    method_vec = [
        float(method),
        float(survival_percent),
        float(cultivated_species),
        float(stems_per_ha),
        float(soil_preparation_type),
        float(clearing),
        float(seed_species),
    ]

    ms.ntrees = ms.dll.regenerate_with_state(
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

    origin = int(req(operation_parameters, "origin"))
    species = int(req(operation_parameters, "species"))
    stems_per_ha = float(req(operation_parameters, "stems_per_ha"))
    height = float(req(operation_parameters, "height"))
    biological_age = float(req(operation_parameters, "biological_age"))
    regen_type = str(req(operation_parameters, "type"))

    # ---- optional ----
    method = int(operation_parameters.get("method", 0))
    breast_height_diameter = operation_parameters.get("breast_height_diameter", None)
    breast_height_age = operation_parameters.get("breast_height_age", None)
    ntrees = operation_parameters.get("ntrees", 10)

    if height <= 0:
        raise MetsiException("Regeneration: Height can not be negative or zero")
    if regen_type not in ("artificial", "natural"):
        raise MetsiException("regeneration 'type' must be 'artificial' or 'natural'")
    if not ntrees or ntrees <= 0:
        raise MetsiException("Parameter 'ntrees' must be positive")
    if stems_per_ha <= 0:
        raise MetsiException("Parameter 'stems_per_ha' must be > 0")

    if regen_type == "artificial":
        stand.artificial_regeneration_year = stand.year

    if getattr(stand, "motti_state", None) is not None:
        if method not in (1, 2, 3):
            raise MetsiException(
                "When Motti is active, regeneration 'method' must be one of: "
                "1=natural, 2=sowing, 3=planting"
            )

        _regeneration_via_motti(
            stand,
            method=method,
            species=species,
            stems_per_ha=stems_per_ha,
            step=int(operation_parameters.get("istep", 0)),
            survival_percent=float(operation_parameters.get("survival_percent", 100.0)),
            soil_preparation_type=int(operation_parameters.get("soil_preparation_type", 0)),
            clearing=int(operation_parameters.get("clearing", 0)),
            seed_tree_species=int(operation_parameters.get("seed_tree_species", 0)),
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
