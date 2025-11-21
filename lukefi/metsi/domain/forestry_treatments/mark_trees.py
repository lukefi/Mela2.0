
import numpy as np
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.data.util.select_units import select_units


def mark_trees(input_: ForestStand, /, **operation_parameters) -> OpTuple[ForestStand]:
    """
    mark_trees treatment (Python equivalent of the R function `ftrt_mark_trees`):

    Selects a subset of reference trees based on a selection profile and *marks* them by
    setting given attributes. If only a part of a tree's stems are selected, the function
    splits that tree into two reference-tree rows: one for the unmarked remainder and a
    new row for the marked stems. If all stems are selected, the attributes are updated
    in-place on that row.

    Parameters (no defaults, required unless noted)
    ----------------------------------------------
    tree_selection : dict
        - Target: {type, var, amount}
        - sets: [ {sfunction, order_var, target_var, target_type, target_amount,
                   profile_x, profile_y, profile_xmode, (optional) profile_xscale}, ... ]
    select_from_all : bool
        Whether per-set selections are computed from original amounts (True) or remaining (False).
    mode : str, optional
        Selection mode forwarded to select_units (e.g. "odds_units").
    attributes : dict
        Mapping of ReferenceTrees field names to values to assign to the *marked* stems,
        e.g. {"tree_type": "SPARE", "management_category": 2}.

    Notes
    -----
    The frequency variable is fixed to 'stems_per_ha' (no configurable freq_var).
    """
    stand = input_
    if stand.reference_trees.size == 0:
        return stand, []

    ts = operation_parameters.get("tree_selection")
    if not ts or "target" not in ts or "sets" not in ts:
        raise MetsiException("Missing 'tree_selection' with 'target' and 'sets'.")

    target = ts["target"]
    sets = ts["sets"]

    select_from_all = operation_parameters.get("select_from_all", True)

    mode = operation_parameters.get("mode", "odds_units")
    attributes = operation_parameters.get("attributes")
    if not attributes or not isinstance(attributes, dict):
        raise MetsiException("Missing 'attributes' (dict of ReferenceTrees fields to set).")

    # Selection amounts for each reference-tree row
    marked_f = select_units(
        context=stand,
        data=stand.reference_trees,
        target_decl=target,
        sets=sets,
        freq_var="stems_per_ha",
        select_from_all=bool(select_from_all),
        mode=str(mode),
    )

    # Work directly on stand.reference_trees.stems_per_ha
    if not hasattr(stand.reference_trees, "stems_per_ha"):
        raise MetsiException("ReferenceTrees is missing 'stems_per_ha' attribute.")

    freq_vec: np.ndarray = stand.reference_trees.stems_per_ha

    # Masks
    all_stems_mask = marked_f == freq_vec
    to_split_mask = (marked_f > 0) & (~all_stems_mask)

    # 1) Rows where ALL stems are marked: set attributes in-place
    all_idxs = np.nonzero(all_stems_mask)[0]
    for idx in all_idxs:
        stand.reference_trees.update(attributes, index=idx)

    # 2) Rows where PARTIAL stems are marked: split
    split_idxs = np.nonzero(to_split_mask)[0]
    if split_idxs.size > 0:
        # Reduce original rows by the marked amount
        if not freq_vec.flags.writeable:
            # make a copy if finalized
            stand.reference_trees.stems_per_ha = freq_vec.copy()
            freq_vec = stand.reference_trees.stems_per_ha

        freq_vec[split_idxs] = freq_vec[split_idxs] - marked_f[split_idxs]

        # Create new rows that carry only the marked stems and desired attributes
        new_rows = []
        start_idx = int(stand.reference_trees.size)

        for offset, idx in enumerate(split_idxs):
            global_idx = start_idx + offset

            row = stand.reference_trees.read(int(idx))
            row["identifier"] = f"{stand.identifier}-{global_idx + 1}-tree"
            row["tree_number"] = global_idx
            row["stems_per_ha"] = marked_f[idx]
            # apply attributes on the *marked* part
            row.update(attributes)
            new_rows.append(row)

        stand.reference_trees.create(new_rows)

    return stand, []
