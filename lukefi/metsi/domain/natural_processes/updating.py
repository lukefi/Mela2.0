from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.sim.treatment import PredeterminedTreatment
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.sim_configuration import TransitionFn
from lukefi.metsi.sim.treatment import Treatment


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

    assert transition is not None
    assert target_year is not None

    if stand.year > target_year:
        raise MetsiException(f"Unable to update stand {stand.identifier} to year {target_year}: "
                             f"already at {stand.year}")

    current = stand
    while current.year < target_year:
        step, treatments = _get_next_step_and_treatments(current, target_year)

        if step > 0:
            current, _ = transition(current, step)

        for treatment in treatments:
            current, _ = treatment(current)

    # Update starting year for relative timepoints
    current.start_time = current.year

    return current, []


def _get_next_step_and_treatments(stand: ForestStand,
                                  target_year: int,
                                  ) -> tuple[int, list[PredeterminedTreatment[ForestStand]]]:
    if stand.predetermined_treatments is not None:
        for treatment in stand.predetermined_treatments:
            if stand.year < treatment.time <= target_year:
                return (treatment.time - stand.year,
                        [treatment_ for treatment_ in stand.predetermined_treatments
                         if treatment_.time == treatment.time])

    return target_year - stand.year, []


update_to_year = Treatment(update_to_year_fn, "update_to_year", default_tags={"initial", "update_to_year"})
