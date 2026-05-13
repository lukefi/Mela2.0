from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.sim_configuration import TransitionFn
from lukefi.metsi.sim.treatment import Treatment


def update_to_year_fn(stand: ForestStand,
                      /,
                      transition: TransitionFn | None = None,
                      target_year: int | None = None
                      ) -> OpTuple[ForestStand]:

    if transition is None:
        raise MetsiException("Required parameter `transition` missing")
    if target_year is None:
        raise MetsiException("Required parameter `target_year` missing")

    current_year = stand.year
    step = target_year - current_year

    if step > 0:
        # update
        return transition(stand, target_year - current_year)

    if step == 0:
        return stand, []

    # Stand is already ahead of requested year
    raise MetsiException(f"Unable to update stand {stand.identifier} to year {target_year}:"
                         f" already at {stand.year}.")


update_to_year = Treatment(update_to_year_fn, "update_to_year", default_tags={"initial", "update_to_year"})
