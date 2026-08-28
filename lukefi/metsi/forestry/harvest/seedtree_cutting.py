from typing import Any

from lukefi.metsi.data.enums.internal import CuttingMethod
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.motti_util import (
    reconcile_reference_trees_from_motti)
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL
from lukefi.metsi.core.collected_data import OpTuple
from lukefi.metsi.core.treatment import Treatment
from lukefi.metsi.forestry.harvest.cutting import cutting_fn
from lukefi.metsi.core.select_units import Mode


def seedtree_cutting_fn(stand: ForestStand,
                        /,
                        seed_tree_class: int = 3,
                        tree_selection: dict[str, Any] | None = None,
                        cutting_method: CuttingMethod | None = None,
                        mode: Mode = Mode.ODDS_UNITS,
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
        _after_seedtree_cutting_in_motti(
            stand,
            seed_tree_class,
        )

    return stand, collected


def _after_seedtree_cutting_in_motti(stand: ForestStand, tree_class: int) -> None:
    """Run Motti seed-tree cutting post-processing and sync Python vectors."""

    ms = stand.motti_state
    if ms is None or ms.yp is None or ms.buffers is None:
        return

    _mark_motti_yp_as_seed_trees(stand, tree_class)

    ms.ntrees = Motti4DLL.after_seedtree_cutting_with_state(
        ms.yy,
        ms.yp,
        int(ms.ntrees),
        ms.buffers,
    )

    reconcile_reference_trees_from_motti(stand)


def _mark_motti_yp_as_seed_trees(stand: ForestStand, tree_class) -> bool:
    """Mark all currently live YP trees as seed trees for Motti.

    Motti's YP vector field at documented index 38 is exposed in the
    CFFI wrapper as ``storie``. For seed-tree cutting the remaining trees
    must have puuluokka/tree class 3 before Motti4AfterSeedtreeCutting is
    called.
    """

    ms = stand.motti_state
    if ms is None or ms.yp is None:
        return False

    changed = False
    for i in range(int(ms.ntrees)):
        t = ms.yp[0][i]
        if float(t.f) <= 0.0:
            continue
        if float(t.storie) != float(tree_class):
            t.storie = float(tree_class)
            changed = True

    return changed


seedtree_cutting = Treatment(seedtree_cutting_fn, "seedtree_cutting")
