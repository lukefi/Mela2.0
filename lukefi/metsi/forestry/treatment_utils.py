from typing import Mapping, Any

import numpy as np
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.sim_configuration import TransitionFn
from lukefi.metsi.sim.treatment import TreatmentFn


def req(params: Mapping[str, Any], name: str) -> Any:
    try:
        return params[name]
    except KeyError as exc:
        raise MetsiException(
            f"Missing required regeneration parameter: '{name}'"
        ) from exc


def prune_zero_stems(func: TreatmentFn[ForestStand] | TransitionFn[ForestStand]
                     ) -> TreatmentFn[ForestStand] | TransitionFn[ForestStand]:
    def prune_zero_stems_wrapper(stand: ForestStand, **parameters) -> OpTuple[ForestStand]:
        new_stand, collected_data = func(stand, **parameters)
        trees = new_stand.reference_trees
        zero_stems_indices = np.nonzero(trees.stems_per_ha == 0)[0]
        trees.delete(zero_stems_indices)

        return new_stand, collected_data

    return prune_zero_stems_wrapper
