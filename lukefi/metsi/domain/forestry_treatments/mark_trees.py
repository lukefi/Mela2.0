
from typing import Any
import numpy as np
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.core.utils.select_units import (
    SelectionSet,
    SelectionTarget,
    select_units,
    Mode)
from lukefi.metsi.core.collected_data import OpTuple
from lukefi.metsi.core.exceptions import MetsiException
from lukefi.metsi.core.treatment import Treatment
from lukefi.metsi.domain.natural_processes.motti_util import split_and_update_motti_yp_tree
from lukefi.metsi.forestry.aggregates_internal import calc_tree_basal_areas
from lukefi.metsi.forestry.storey import manage_existing_storeys


def mark_trees_fn(input_: ForestStand,
                  /,
                  tree_selection: dict[str, Any] | None = None,
                  select_from_all: bool = True,
                  mode: Mode = Mode.ODDS_UNITS,
                  attributes: dict[str, Any] | None = None) -> OpTuple[ForestStand]:
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

    ts = tree_selection
    if not ts or "target" not in ts or "sets" not in ts:
        raise MetsiException("Missing 'tree_selection' with 'target' and 'sets'.")

    target: SelectionTarget = ts["target"]
    sets: list[SelectionSet] = ts["sets"]

    if not attributes:
        raise MetsiException("Missing 'attributes' (dict of ReferenceTrees fields to set).")

    # Selection amounts for each reference-tree row
    marked_f = select_units(
        context=stand,
        data=stand.reference_trees,
        target_decl=target,
        sets=sets,
        freq_var="stems_per_ha",
        select_from_all=bool(select_from_all),
        mode=mode,
    )

    # Work directly on stand.reference_trees.stems_per_ha
    if not hasattr(stand.reference_trees, "stems_per_ha"):
        raise MetsiException("ReferenceTrees is missing 'stems_per_ha' attribute.")

    f0 = stand.reference_trees.stems_per_ha

    # Masks
    all_stems_mask = marked_f == f0
    to_split_mask = (marked_f > 0) & (~all_stems_mask)

    # 1) Rows where ALL stems are marked: set attributes in-place
    all_idxs = np.nonzero(all_stems_mask)[0]
    for idx in all_idxs:
        stand.reference_trees.update(attributes, index=int(idx))

    # 2) Rows where PARTIAL stems are marked: split
    split_idxs = np.nonzero(to_split_mask)[0]
    if split_idxs.size > 0:

        f_diff = f0 - marked_f

        # Create new rows that carry only the marked stems and desired attributes
        new_rows = []
        start_idx = int(max(stand.reference_trees.tree_number)) + 1

        for offset, idx in enumerate(split_idxs):
            new_tree_number = start_idx + offset
            row = stand.reference_trees.read(int(idx))

            if stand.motti_state is not None and not row["sapling"]:
                # Updating marked trees into Motti
                split_and_update_motti_yp_tree(stand.motti_state,
                                               new_tree_number,
                                               int(row["tree_number"]),
                                               int(row["stratum"]),
                                               float(marked_f[idx]),
                                               float(f_diff[idx]),
                                               attributes.get("storey"))
            # Resolve if sapling also splits
            is_splitted = True
            if stand.motti_state and row["sapling"]:
                # Saplings in motti context never split
                is_splitted = False
                if marked_f[idx] > f_diff[idx]:
                    # update only attributes for original sapling if marked stem count exceeds 50% of original
                    stand.reference_trees.update(attributes, int(idx))

            if is_splitted:
                # Making a new reference tree
                row["identifier"] = f"{stand.identifier}-{new_tree_number}-tree"
                row["tree_number"] = new_tree_number
                row["stems_per_ha"] = marked_f[idx]
                # apply attributes on the *marked* part
                row.update(attributes)
                new_rows.append(row)
                # Update original reference tree stems
                if not f0.flags.writeable:
                    f0 = f0.copy()
                    f0.flags.writeable = True
                f0[idx] = f_diff[idx]

        stand.reference_trees.stems_per_ha = f0
        stand.reference_trees.create(new_rows)

    trees = stand.reference_trees

    # Pre-calculate basal area for all trees
    trees.basal_area = calc_tree_basal_areas(trees.breast_height_diameter)

    # TODO: Should user-marked storeys get special treatment?
    manage_existing_storeys(trees)

    return stand, []


mark_trees = Treatment(mark_trees_fn, "mark_trees")
