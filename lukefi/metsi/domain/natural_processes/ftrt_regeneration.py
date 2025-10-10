from __future__ import annotations
from typing import Optional
from lukefi.metsi.data.model import ForestStand, TreeStratum
from lukefi.metsi.sim.collected_data import OpTuple

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

    # Create a new stratum (matches the spirit of the R code)
    ts = TreeStratum()
    ts.stand = stand
    ts.origin = origin
    ts.species = species
    ts.stems_per_ha = f
    ts.mean_height = Hgm
    ts.mean_diameter = Dgm
    ts.biological_age = age_biol
    # other fields left default/zero; downstream growth will fill as needed

    if not hasattr(stand, "tree_strata") or stand.tree_strata is None:
        stand.tree_strata = []
    stand.tree_strata.append(ts)

    cdata.store("regeneration", {
        "time": sim_time,
        "labels": (labels or []) + ["regeneration"],
        "origin": origin,
        "species": species,
        "stems_per_ha": f,
        "Hgm": Hgm,
        "Dgm": Dgm,
        "age_biol": age_biol,
    })
    return (stand, cdata)
