from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.treatment import Treatment
from lukefi.metsi.forestry.harvest.cutting import cutting_fn
from lukefi.metsi.domain.natural_processes.grow_motti import (
    after_seedtree_cutting_in_motti,
)


def seedtree_cutting_fn(input_: ForestStand, /, **operation_parameters) -> OpTuple[ForestStand]:
    """
    Seed-tree cutting treatment.

    First performs the normal cutting treatment, including ReferenceTrees -> YP
    stem-count reduction when Motti is used. Then marks all remaining YP trees as tree class 3 and
    calls Motti4AfterSeedtreeCutting so Motti can create natural regeneration
    into the sapling vector.
    """
    stand, collected = cutting_fn(input_, **operation_parameters)

    if getattr(stand, "motti_state", None) is not None:
        after_seedtree_cutting_in_motti(
            stand,
            tree_class=int(operation_parameters.get("seed_tree_class", 3)),
        )

    return stand, collected


seedtree_cutting = Treatment(seedtree_cutting_fn, "seedtree_cutting")
