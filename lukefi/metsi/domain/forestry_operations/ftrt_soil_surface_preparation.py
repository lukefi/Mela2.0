from typing import Optional
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple

def ftrt_soil_surface_preparation(
    op: OpTuple[ForestStand],
    /,
    **operation_parameters
) -> OpTuple[ForestStand]:
    stand, cdata = op

    method = operation_parameters.get("method")
    intensity = operation_parameters.get("intensity")
    labels = (operation_parameters.get("labels") or []) + ["soil_surface_preparation"]

    stand, cdata = op
    sim_year: Optional[int] = stand.year
    stand.soil_surface_preparation_year = sim_year
    cdata.store("soil_surface_preparation", {
        "time": sim_year,
        "labels": labels,
        "method": method,
        "intensity_per_ha": intensity,
    })
    return (stand, cdata)
