from functools import wraps
from typing import Mapping, Any
import numpy as np
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.transition import TransitionFn
from lukefi.metsi.sim.treatment import TreatmentFn


def req(params: Mapping[str, Any], name: str) -> Any:
    try:
        return params[name]
    except KeyError as exc:
        raise MetsiException(
            f"Missing required regeneration parameter: '{name}'"
        ) from exc


def prune_zero_stems_treatment(func: TreatmentFn[ForestStand]) -> TreatmentFn[ForestStand]:
    @wraps(func)
    def prune_zero_stems_treatment_wrapper(stand: ForestStand, **parameters) -> OpTuple[ForestStand]:
        new_stand, collected_data = func(stand, **parameters)
        trees = new_stand.reference_trees
        zero_stems_indices = np.nonzero(trees.stems_per_ha == 0)[0]
        trees.delete(zero_stems_indices)

        return new_stand, collected_data

    return prune_zero_stems_treatment_wrapper


def prune_zero_stems_transition(func: TransitionFn[ForestStand]) -> TransitionFn[ForestStand]:
    @wraps(func)
    def prune_zero_stems_transition_wrapper(stand: ForestStand, step: int, **parameters) -> OpTuple[ForestStand]:
        new_stand, collected_data = func(stand, step, **parameters)
        trees = new_stand.reference_trees
        zero_stems_indices = np.nonzero(trees.stems_per_ha == 0)[0]
        trees.delete(zero_stems_indices)

        return new_stand, collected_data

    return prune_zero_stems_transition_wrapper
