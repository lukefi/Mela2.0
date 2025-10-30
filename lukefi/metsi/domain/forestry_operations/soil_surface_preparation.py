from typing import Optional
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple

def soil_surface_preparation(
    op: ForestStand,
    /,
    **_operation_parameters
) -> OpTuple[ForestStand]:

    """
    Simulate soil surface preparation on a stand (e.g., mounding).

    Side effects:
      - Sets `stand.soil_surface_preparation_year` to the current simulation year.

    Parameters
    ----------
    op : ForestStand
        The stand to update.

    Returns
    -------
    OpTuple[ForestStand]
        The updated (stand, collected_data) tuple.

    """

    stand = op

    sim_year: Optional[int] = stand.year
    stand.soil_surface_preparation_year = sim_year

    return stand, []
