
import numpy as np
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.data.util.select_units import select_units, SelectionSet, SelectionTarget


def mark_trees(input_: ForestStand, /, **operation_parameters) -> OpTuple[ForestStand]:
    """
    mark_trees treatment:

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
    freq_var : str
        Frequency column name in ReferenceTrees (e.g. "stems_per_ha").
    select_from_all : bool
        Whether per-set selections are computed from original amounts (True) or remaining (False).
    mode : str, optional
        Selection mode forwarded to select_units (e.g. "odds_units").
    attributes : dict
        Mapping of ReferenceTrees field names to values to assign to the *marked* stems,
        e.g. {"tree_type": "retained", "management_category": 2}.

    Returns
    -------
    (ForestStand, list)
        Modified stand and empty collected-data list.
    """
    stand = input_
    if stand.reference_trees.size == 0:
        return stand, []


    ts = operation_parameters.get("tree_selection")
    if not ts or "Target" not in ts or "sets" not in ts:
        raise MetsiException("Missing 'tree_selection' with 'Target' and 'sets'.")

    target = ts["Target"]
    for k in ("type", "var", "amount"):
        if k not in target:
            raise MetsiException(f"tree_selection.Target missing '{k}'.")

    # Global target
    target_decl = SelectionTarget()
    target_decl.type = target["type"]
    target_decl.var = target["var"]
    target_decl.amount = target["amount"]

    # Sets
    sets_in = ts["sets"]
    if not isinstance(sets_in, (list, tuple)) or len(sets_in) == 0:
        raise MetsiException("tree_selection.sets must be a non-empty list.")

    py_sets: list[SelectionSet[ForestStand, ReferenceTrees]] = []
    for i, s in enumerate(sets_in):
        for req in ("sfunction", "order_var", "target_var", "target_type", "target_amount",
                    "profile_x", "profile_y", "profile_xmode"):
            if req not in s:
                raise MetsiException(f"sets[{i}] missing '{req}'.")
        ss = SelectionSet[ForestStand, ReferenceTrees]()
        ss.sfunction = s["sfunction"]
        ss.order_var = s["order_var"]
        ss.target_var = s["target_var"]
        ss.target_type = s["target_type"]
        ss.target_amount = s["target_amount"]
        ss.profile_x = np.asarray(s["profile_x"], dtype=np.float64)
        ss.profile_y = np.asarray(s["profile_y"], dtype=np.float64)
        ss.profile_xmode = s["profile_xmode"]
        ss.profile_xscale = s.get("profile_xscale")
        if ss.profile_x.shape != ss.profile_y.shape or ss.profile_x.ndim != 1 or ss.profile_x.size < 2:
            raise MetsiException(f"sets[{i}]: profile_x/profile_y must be 1D arrays of equal length (>=2).")
        py_sets.append(ss)

    # Required parameters
    freq_var = operation_parameters.get("freq_var")
    if not freq_var:
        raise MetsiException("Missing 'freq_var' (e.g., 'stems_per_ha').")
    select_from_all = operation_parameters.get("select_from_all")
    if select_from_all is None:
        raise MetsiException("Missing 'select_from_all' (bool).")
    mode = operation_parameters.get("mode")
    attributes = operation_parameters.get("attributes")
    if not attributes or not isinstance(attributes, dict):
        raise MetsiException("Missing 'attributes' (dict of ReferenceTrees fields to set).")

    # Selection amounts for each reference-tree row
    marked_f = select_units(
        context=stand,
        data=stand.reference_trees,
        target_decl=target_decl,
        sets=py_sets,
        freq_var=freq_var,
        select_from_all=bool(select_from_all),
        mode=str(mode),
    )

    if not hasattr(stand.reference_trees, freq_var):
        raise MetsiException(f"Unknown freq_var '{freq_var}' in ReferenceTrees.")

    freq_vec: np.ndarray = getattr(stand.reference_trees, freq_var)
    if np.any(marked_f > freq_vec):
        raise MetsiException("mark_trees would mark more stems than available; fix targets/profiles.")

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
            setattr(stand.reference_trees, freq_var, freq_vec.copy())
            freq_vec = getattr(stand.reference_trees, freq_var)
        freq_vec[split_idxs] = freq_vec[split_idxs] - marked_f[split_idxs]

        # Create new rows that carry only the marked stems and desired attributes
        new_rows = []
        for idx in split_idxs:
            row = stand.reference_trees.read(idx)
            row[freq_var] = marked_f[idx]
            # apply attributes on the *marked* part
            row.update(attributes)
            new_rows.append(row)

        stand.reference_trees.create(new_rows)

    return stand, []
