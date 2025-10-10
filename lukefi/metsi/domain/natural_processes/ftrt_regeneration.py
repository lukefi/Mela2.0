from __future__ import annotations
from typing import Optional, Any
import numpy as np

from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.vector_model import TreeStrata as SoATreeStrata 

def ftrt_regeneration(
    op: OpTuple[ForestStand],
    *,
    origin: int,
    species: int,
    f: float,
    Hgm: float,
    Dgm: float,
    age_biol: float,
    labels: list[str] | None = None,
    sim_time: Optional[int] = None,
) -> OpTuple[ForestStand]:
    stand, cdata = op

    strata = getattr(stand, "tree_strata", None)
    if strata is None:
        raise MetsiException("stand.tree_strata missing")

    if isinstance(strata, SoATreeStrata) or hasattr(strata, "create"):
        new_row = {
            # identifier can be blank or something like f"{stand.identifier}_regen_{sim_time or 0}"
            "identifier": "",
            "species": int(species),
            "origin": int(origin),
            "stems_per_ha": float(f),
            "mean_height": float(Hgm),
            "mean_diameter": float(Dgm),
            "biological_age": float(age_biol),
            # useful bookkeeping fields
            "cutting_year": int(sim_time) if sim_time is not None else -1,
            "sapling_stratum": True,
            "sapling_stems_per_ha": float(f),
            # everything else will get sensible defaults (NaN/0/empty) in VectorData.defaultify
        }
        strata.create(new_row)

    cdata.store("regeneration", {
        "time": sim_time,
        "labels": (labels or []) + ["regeneration"],
        "params": {"origin": origin, "species": species, "f": f, "Hgm": Hgm, "Dgm": Dgm, "age_biol": age_biol},
    })
    return (stand, cdata)
