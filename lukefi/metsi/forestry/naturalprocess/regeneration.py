from typing import Any
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.app.utils import MetsiException

def regeneration(input_: OpTuple[ForestStand], /, **operation_parameters) -> OpTuple[ForestStand]:
    """
    Regeneration treatment: add *reference trees*.
    - No cdata collection by design.
    - Parameters (all via **operation_parameters):
        origin: int                 # e.g. 2 (planted)
        method: Optional[int]       # accepted, unused (present for parity with R)
        species: int                # tree species code
        stems_per_ha: float         # total stems/ha to distribute to created trees
        height: float               # initial height (m)
        biological_age: float       # biological age (years)
        breast_height_diameter: Optional[float] = None
        breast_height_age: Optional[float] = None
        ntrees: Optional[int] = 10  # number of reference trees to create
        labels: Optional[list[str]] = None  # accepted, unused
        type: str = "artificial"            # "artificial" | "natural"
    """
    stand, cdata = input_

    # ---- required ----
    def req(name: str):
        if name not in operation_parameters:
            raise MetsiException(f"Missing required regeneration parameter: '{name}'")
        return operation_parameters[name]

    origin = int(req("origin"))
    species = int(req("species"))
    stems_per_ha = float(req("stems_per_ha"))
    height = float(req("height"))
    biological_age = float(req("biological_age"))

    # ---- optional ----
    breast_height_diameter = operation_parameters.get("breast_height_diameter")
    breast_height_age = operation_parameters.get("breast_height_age")
    ntrees = int(operation_parameters.get("ntrees", 10))
    _method = operation_parameters.get("method")
    _labels = operation_parameters.get("labels")
    regen_type = str(operation_parameters.get("type", "artificial")).lower()

    if regen_type not in ("artificial", "natural"):
        raise MetsiException("regeneration 'type' must be 'artificial' or 'natural'")
    if ntrees <= 0:
        raise MetsiException("Parameter 'ntrees' must be positive")
    if stems_per_ha <= 0:
        raise MetsiException("Parameter 'stems_per_ha' must be > 0")

    if regen_type == "artificial":
        stand.artificial_regeneration_year = stand.year

    # ---- create trees ----
    per_tree_stems = stems_per_ha / float(ntrees)
    if height <= 0:
        height = 0.01  # clamp to avoid downstream divide-by-zero
    new_rows = []
    for _ in range(ntrees):
        new_rows.append({
            "identifier": "",
            "species": species,
            "origin": origin,
            "stems_per_ha": per_tree_stems,
            "height": height,
            "measured_height": height,  # helpful for growth models that prefer measured height
            "biological_age": biological_age,
            "breast_height_diameter": None if breast_height_diameter is None else float(breast_height_diameter),
            "breast_height_age": None if breast_height_age is None else float(breast_height_age),
        })
    stand.reference_trees.create(new_rows)

    # No cdata here by design
    return (stand, cdata)
