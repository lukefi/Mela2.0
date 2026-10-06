import numpy as np

from lukefi.metsi.data.vector_model import ReferenceTrees

def collect_removed_trees(rts_original: ReferenceTrees, rts_modified: ReferenceTrees) -> ReferenceTrees:
    """ Collect trees that have been removed or reduced in stem count. """
    id1 = rts_original.tree_number
    f1 = rts_original.stems_per_ha
    id2 = rts_modified.tree_number
    f2 = rts_modified.stems_per_ha
    # Trees that have have been reduced in stem count
    id_common, idx1, idx2 = np.intersect1d(
        id1,
        id2,
        return_indices=True
    )
    f_diff = f1[idx1] - f2[idx2]
    f_diff_mask = f_diff > 0.0
    id_changed = id_common[f_diff_mask]
    f_diff_changed = f_diff[f_diff_mask]

    # Trees that have been completely removed
    id_removed = np.setdiff1d(id1, id2)

    # Compose the removed trees into a new ReferenceTrees object
    removed_trees: ReferenceTrees = rts_original[np.isin(id1, np.concatenate([id_changed, id_removed]))]
    removed_trees.stems_per_ha[np.isin(removed_trees.tree_number, id_changed)] = f_diff_changed

    return removed_trees
