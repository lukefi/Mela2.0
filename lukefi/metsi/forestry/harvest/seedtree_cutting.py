from typing import Any

from lukefi.metsi.data.enums.internal import CuttingMethod
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.treatment import Treatment
from lukefi.metsi.forestry.harvest.cutting import cutting_fn
from lukefi.metsi.domain.natural_processes.motti_util import (
    after_seedtree_cutting_in_motti,
)


def seedtree_cutting_fn(stand: ForestStand,
                        /,
                        seed_tree_class: int = 3,
                        tree_selection: dict[str, Any] | None = None,
                        cutting_method: CuttingMethod | None = None,
                        mode: str = "odds_units",
                        select_from_all: bool = False
                        ) -> OpTuple[ForestStand]:
    """
    Seed-tree cutting treatment.

    First performs the normal cutting treatment, including ReferenceTrees -> YP
    stem-count reduction when Motti is used. Then marks all remaining YP trees as tree class 3 and
    calls Motti4AfterSeedtreeCutting so Motti can create natural regeneration
    into the sapling vector.
    """
    stand, collected = cutting_fn(stand,
                                  tree_selection=tree_selection,
                                  cutting_method=cutting_method,
                                  mode=mode,
                                  select_from_all=select_from_all)

    if stand.motti_state is not None:
        after_seedtree_cutting_in_motti(
            stand,
            tree_class=seed_tree_class,
        )

    return stand, collected


seedtree_cutting = Treatment(seedtree_cutting_fn, "seedtree_cutting")
