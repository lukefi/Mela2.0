import re
import numpy as np
import numpy.typing as npt
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees


def _next_reference_tree_number(rt: ReferenceTrees) -> int:
    vals = []
    for v in rt.tree_number.tolist():
        try:
            iv = int(v)
            if iv > 0:
                vals.append(iv)
        except (TypeError, ValueError):
            pass
    return (max(vals) + 1) if vals else 1


def _next_reference_tree_identifier_suffix(stand: ForestStand) -> int:
    rt = stand.reference_trees
    used = set()

    suffix_re = re.compile(rf"^{re.escape(stand.identifier)}-(\d+)-tree$")

    for ident in rt.identifier.tolist():
        s = str(ident)
        m = suffix_re.match(s)
        if m:
            try:
                used.add(int(m.group(1)))
            except ValueError:
                pass

    n = 1
    while n in used:
        n += 1
    return n


def new_reference_tree_identity(stand: ForestStand) -> tuple[str, int]:
    """ Composes a new identifier and tree number value for a new reference tree
            keeping tree number as unique and incremental within the stand.
    """
    rt = stand.reference_trees
    tree_number = np.max(rt.tree_number).item() if rt.size > 0 else 0
    tree_number += 1
    identifier = f"{stand.identifier}-{tree_number}-tree"
    return identifier, tree_number


def update_stand_growth(stand: ForestStand,
                        diameters: npt.NDArray[np.float64],
                        heights: npt.NDArray[np.float64],
                        stems: npt.NDArray[np.float64],
                        step: int,
                        update_sapling: bool = True):
    """In-place update stand's reference trees with given diameters, heights and stem count.
    Increase ages for trees and stand. Remove sapling flag from trees that have grown beyond 1.3m. """

    trees = stand.reference_trees

    trees.biological_age = trees.biological_age + step
    trees.breast_height_age = np.where(
        (trees.height < 1.3) & (1.3 <= heights),
        trees.biological_age,
        trees.breast_height_age)
    trees.breast_height_diameter = diameters
    trees.height = heights
    trees.stems_per_ha = stems

    if update_sapling:
        trees.sapling = np.where(
            trees.height >= 1.3,
            False,
            trees.sapling)

    stand.year = (stand.year or 0) + step
