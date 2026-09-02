from typing import Optional
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.core.collected_data import OpTuple
from lukefi.metsi.core.treatment import Treatment


def soil_surface_preparation_fn(stand: ForestStand) -> OpTuple[ForestStand]:
    """
    Simulate soil surface preparation on a stand (e.g., mounding).

    Side effects:
      - Sets `stand.soil_surface_preparation_year` to the current simulation year.

    Parameters
    ----------
    stand : ForestStand
        The stand to update.

    Returns
    -------
    OpTuple[ForestStand]
        The updated (stand, collected_data) tuple.

    """

    sim_year: Optional[int] = stand.year
    stand.soil_surface_preparation_year = sim_year

    return stand, []


soil_surface_preparation = Treatment(soil_surface_preparation_fn, "soil_surface_preparation")
