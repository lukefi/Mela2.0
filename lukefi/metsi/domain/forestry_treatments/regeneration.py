from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.forestry.treatment_utils import req

def regeneration(input_: ForestStand, /, **operation_parameters) -> OpTuple[ForestStand]:
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
    """
    stand = input_


    origin = int(req(operation_parameters, "origin"))
    species = int(req(operation_parameters, "species"))
    stems_per_ha = float(req(operation_parameters, "stems_per_ha"))
    height = float(req(operation_parameters, "height"))
    biological_age = float(req(operation_parameters, "biological_age"))
    regen_type = str(req(operation_parameters, "type"))

    # ---- optional ----
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

    # ---- create trees ----
    per_tree_stems = stems_per_ha / float(ntrees)
    start_idx = int(stand.reference_trees.size)
    new_rows = []
    for i in range(ntrees):
        global_idx = start_idx + i
        identifier = f"{stand.identifier}-{global_idx + 1}-tree"
        new_rows.append({
            "identifier": identifier,
            "tree_number": global_idx,
            "species": species,
            "origin": origin,
            "stems_per_ha": per_tree_stems,
            "height": height,
            "biological_age": biological_age,
            "breast_height_diameter": None if breast_height_diameter is None else float(breast_height_diameter),
            "breast_height_age": None if breast_height_age is None else float(breast_height_age),
        })
    stand.reference_trees.create(new_rows)

    # No cdata here by design
    return stand, []
