from copy import copy

from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.collected_data import NaturalProcessInfo
from lukefi.metsi.sim.collected_data import CollectedData
from lukefi.metsi.sim.sim_configuration import TransitionFn


def natural_process_transition(natural_process_func: TransitionFn[ForestStand]):
    """Wrap a natural process transition function so that NaturalProcessInfo is collected.

    Args:
        natural_process_func (TransitionFn[ForestStand]): A natural process transition function, e.g.
            `grow_motti_dll_fn`.
    """
    def wrapper(computational_unit: ForestStand, step: int,  **params) -> tuple[ForestStand, list[CollectedData]]:
        np_info = NaturalProcessInfo()
        np_info.start_year = computational_unit.year
        np_info.trees_before = copy(computational_unit.reference_trees)
        retval = natural_process_func(computational_unit, step, **params)
        np_info.step = retval[0].time - np_info.start_year
        np_info.trees_after = copy(retval[0].reference_trees)
        retval[1].append(np_info)
        return retval
    return wrapper
