from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.sim.transition import TransitionFn
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.treatment import Treatment
from lukefi.metsi.sim.updating import get_step_and_treatments


def update_to_year_fn(stand: ForestStand,
                      /,
                      transition: TransitionFn[ForestStand] | None = None,
                      target_year: int | None = None
                      ) -> OpTuple[ForestStand]:
    """
        Apply a transition function to update the stand to target year.

        If the stand has predetermined treatments, they will be applied at the appropriate time points.
        In such a case the transition can take place in several discrete steps (i.e. the time intervals between
        the treatments).
    """

    assert transition is not None, "required parameter `transition` missing "
    assert target_year is not None, "required parameter `target_year` missing"

    if stand.time > target_year:
        raise MetsiException(f"Unable to update stand {stand.identifier} to year {target_year}: "
                             f"already at {stand.time}")

    current = stand
    keep_running = True
    while keep_running:
        step, treatments = get_step_and_treatments(current, target_year)

        for treatment in treatments:
            current, _ = treatment(current)

        if step > 0:
            current, _ = transition(current, step)
        else:
            keep_running = False

    # Update starting year for relative timepoints
    current.start_time = current.time

    return current, []


update_to_year = Treatment(update_to_year_fn, "update_to_year", default_tags={"initial", "update_to_year"})
