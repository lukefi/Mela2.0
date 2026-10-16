from typing import Optional
from lukefi.metsi.data.enums.internal import SoilPreparationType
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.core.collected_data import OpTuple
from lukefi.metsi.core.treatment import Treatment


def soil_surface_preparation_fn(stand: ForestStand,
                                /,
                                soil_preparation_param: SoilPreparationType | None = None
                                )-> OpTuple[ForestStand]:
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
    assert soil_preparation_param, "soil_preparation_param must be provided for soil surface preparation treatment"

    sim_year: Optional[int] = stand.year
    stand.soil_surface_preparation_year = sim_year
    stand.soil_preparation_type = soil_preparation_param

    return stand, []


soil_surface_preparation = Treatment(soil_surface_preparation_fn, "soil_surface_preparation")
