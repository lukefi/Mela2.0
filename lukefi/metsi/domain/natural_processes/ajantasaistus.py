from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.sim_configuration import Transition
from lukefi.metsi.sim.treatment import Treatment


def ajantasaistus_fn(stand: ForestStand,
                     /,
                     **params
                     ) -> OpTuple[ForestStand]:
    transition: Transition[ForestStand] = params["transition"]
    target_year: int = params["target_year"]
    current_year = stand.year
    if current_year < target_year:
        # update
        return transition(stand, target_year - current_year)
    elif current_year == target_year:
        return stand, []
    else:
        # Stand is already ahead of requested year
        raise MetsiException(f"Unable to update stand {stand.identifier} to year {target_year}: already at {stand.year}.")

ajantasaistus = Treatment(ajantasaistus_fn, "ajantasaistus")
