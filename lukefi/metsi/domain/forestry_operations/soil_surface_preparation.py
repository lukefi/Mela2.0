from typing import Optional
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple

def soil_surface_preparation(
    op: OpTuple[ForestStand],
    /,
    **operation_parameters
) -> OpTuple[ForestStand]:

    """
    Simulate soil surface preparation on a stand (e.g., mounding).

    Side effects:
      - Sets `stand.soil_surface_preparation_year` to the current simulation year.
      - Appends a record to collected data under key "soil_surface_preparation":
        {
          "time": <year>,
          "labels": <labels + ["soil_surface_preparation"]>,
          "method": <str | None>,
          "intensity_per_ha": <float | None>
        }

    Parameters
    ----------
    op : OpTuple[ForestStand]
        The (stand, collected_data) tuple to update.
    **operation_parameters
        method : str | None
            Preparation method (e.g., "mounding").
        intensity : float | None
            Treatment intensity per hectare.
        labels : list[str] | None
            Additional labels to include. "soil_surface_preparation" is always added.

    Returns
    -------
    OpTuple[ForestStand]
        The updated (stand, collected_data) tuple.

    """

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
