from typing import Optional
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple

def ftrt_soil_surface_preparation(
    op: OpTuple[ForestStand],
    *,
    method: str | None = None,
    intensity: float | None = None,
    labels: list[str] | None = None,
) -> OpTuple[ForestStand]:
    stand, cdata = op
    sim_year: Optional[int] = stand.year
    stand.soil_surface_preparation_year = sim_year
    cdata.store("soil_surface_preparation", {
        "time": sim_year,
        "labels": (labels or []) + ["soil_surface_preparation"],
        "method": method,
        "intensity_per_ha": intensity,
    })
    return (stand, cdata)
