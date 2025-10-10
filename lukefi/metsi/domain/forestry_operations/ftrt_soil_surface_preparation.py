# lukefi/metsi/domain/forestry_operations/r_ported/soil_surface_preparation.py
from __future__ import annotations
from typing import Optional
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple

def ftrt_soil_surface_preparation(
    op: OpTuple[ForestStand],
    *,
    method: str | None = None,
    intensity: float | None = None,
    labels: list[str] | None = None,
    sim_time: Optional[int] = None,
) -> OpTuple[ForestStand]:
    stand, cdata = op
    # store to stand (simple metadata, allowed)
    setattr(stand, "soil_surface_preparation_year", sim_time)
    cdata.store("soil_surface_preparation", {
        "time": sim_time,
        "labels": (labels or []) + ["soil_surface_preparation"],
        "method": method,
        "intensity_per_ha": intensity,
    })
    return (stand, cdata)
