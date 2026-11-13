
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


def _mark_fully_selected_trees(stand: ForestStand,
                               selected_indices: np.ndarray,
                               attributes: dict[str, Any]) -> None:
    for idx in selected_indices:
        stand.reference_trees.update(attributes, index=int(idx))


def _update_motti_for_partial_mark(stand: ForestStand,
                                   index: int,
                                   row: dict[str, Any],
                                   new_tree_number: int,
                                   marked_f: float,
                                   remaining_f: float,
                                   attributes: dict[str, Any]) -> bool:
    """Update Motti for a partial mark and return whether the reference tree splits."""
    motti_state = stand.motti_state
    if motti_state is None:
        return True

    if not row["sapling"]:
        split_and_update_motti_yp_tree(
            motti_state,
            new_tree_number,
            int(row["tree_number"]),
            int(row["stratum"]),
            marked_f,
            remaining_f,
            attributes.get("storey"),
        )
        return True

    # Motti saplings never split; mark the original only when most stems are selected.
    if marked_f > remaining_f:
        stand.reference_trees.update(attributes, index=index)
    return False


def _split_partially_selected_trees(stand: ForestStand,
                                    marked_f: np.ndarray,
                                    split_indices: np.ndarray,
                                    attributes: dict[str, Any]) -> None:
    trees = stand.reference_trees
    remaining_f = trees.stems_per_ha - marked_f
    start_tree_number = int(max(trees.tree_number)) + 1
    new_rows = []

    for offset, raw_idx in enumerate(split_indices):
        idx = int(raw_idx)
        new_tree_number = start_tree_number + offset
        row = trees.read(idx)
        if not _update_motti_for_partial_mark(
            stand,
            idx,
            row,
            new_tree_number,
            float(marked_f[idx]),
            float(remaining_f[idx]),
            attributes,
        ):
            continue

        row["identifier"] = f"{stand.identifier}-{new_tree_number}-tree"
        row["tree_number"] = new_tree_number
        row["stems_per_ha"] = marked_f[idx]
        row.update(attributes)
        new_rows.append(row)
        trees.update({"stems_per_ha": remaining_f[idx]}, index=idx)

    if new_rows:
        trees.create(new_rows)


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

    all_stems_mask = marked_f == f0
    to_split_mask = (marked_f > 0) & (~all_stems_mask)

    _mark_fully_selected_trees(
        stand,
        np.nonzero(all_stems_mask)[0],
        attributes,
    )
    split_indices = np.nonzero(to_split_mask)[0]
    if split_indices.size > 0:
        _split_partially_selected_trees(stand, marked_f, split_indices, attributes)

    trees = stand.reference_trees

    # Pre-calculate basal area for all trees
    trees.basal_area = calc_tree_basal_areas(trees.breast_height_diameter)

    # TODO: Should user-marked storeys get special treatment?
    manage_existing_storeys(trees)

    return stand, []


mark_trees = Treatment(mark_trees_fn, "mark_trees")
