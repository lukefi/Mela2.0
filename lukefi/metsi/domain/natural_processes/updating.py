from lukefi.metsi.sim.exceptions import MetsiException
from lukefi.metsi.sim.model import ComputationalUnit
from lukefi.metsi.sim.transition import TransitionFn
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.treatment import Treatment
from lukefi.metsi.sim.updating import get_step_and_treatments


def update_to_year_fn[T: ComputationalUnit](unit: T,
                                            /,
                                            transition: TransitionFn[T] | None = None,
                                            target_year: int | None = None,
                                            **transition_params
                                            ) -> OpTuple[T]:
    """
        Apply a transition function to update the stand to target year.

        If the stand has predetermined treatments, they will be applied at the appropriate time points.
        In such a case the transition can take place in several discrete steps (i.e. the time intervals between
        the treatments).
    """

    assert transition is not None, "required parameter `transition` missing "
    assert target_year is not None, "required parameter `target_year` missing"

    if unit.time > target_year:
        raise MetsiException(f"Unable to update stand {unit.identifier} to year {target_year}: "
                             f"already at {unit.time}")

    current = unit
    keep_running = True
    while keep_running:
        step, treatments = get_step_and_treatments(current, target_year)

        for treatment in treatments:
            current, _ = treatment(current)

        if step > 0:
            current, _ = transition(current, step, **transition_params)
        else:
            keep_running = False

    # Update starting year for relative timepoints
    current.start_time = current.time

    return current, []


update_to_year = Treatment[ComputationalUnit](update_to_year_fn, "update_to_year", default_tags={"initial", "update_to_year"})
