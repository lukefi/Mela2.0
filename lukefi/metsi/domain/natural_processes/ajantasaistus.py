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
    step = target_year - current_year

    if step > transition.max_step:
        raise MetsiException(f"Requested transition step size ({step}) is "
                             f"larger than the maximum step size ({transition.max_step})")

    if step > 0:
        # update
        return transition(stand, target_year - current_year)

    if step == 0:
        return stand, []

    # Stand is already ahead of requested year
    raise MetsiException(f"Unable to update stand {stand.identifier} to year {target_year}:"
                         f" already at {stand.year}.")


ajantasaistus = Treatment(ajantasaistus_fn, "ajantasaistus", default_tags={"initial", "ajantasaistus"})
