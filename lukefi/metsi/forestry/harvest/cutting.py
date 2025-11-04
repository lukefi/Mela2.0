from typing import Optional
import numpy as np
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.data.util.select_units import select_units, SelectionSet, SelectionTarget


def cutting(input_: ForestStand, /, **operation_parameters) -> OpTuple[ForestStand]:
    """
    cutting treatment:
      - Requires operation_parameters['tree_selection'] with:
          Target: {type, var, amount}
          sets: [ {sfunction, order_var, target_var, target_type, target_amount,
                   profile_x, profile_y, profile_xmode, (optional) profile_xscale}, ... ]
      - Applies removals to stand.reference_trees.stems_per_ha.
      - Updates stand.cutting_year and stand.method_of_last_cutting if provided.
    """
    stand = input_
    if stand.reference_trees.size == 0:
        return stand, []
    trees: ReferenceTrees = stand.reference_trees  # direct access as you prefer
    if not isinstance(trees, ReferenceTrees):
        raise MetsiException("cutting requires stand.reference_trees (ReferenceTrees SoA).")

    prereq = operation_parameters.get("prerequisite")
    if prereq is not None:
        filled = False
        # Try (stand, trees) first, then (stand), finally treat as bool
        try:
            filled = bool(prereq(stand, trees))
        except TypeError:
            try:
                filled = bool(prereq(stand))
            except TypeError:
                filled = bool(prereq)
        if not filled:
            # Cutting can not be done
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

    # Required/explicit params (no defaults here)
    freq_var = operation_parameters.get("freq_var", "stems_per_ha")
    if not freq_var:
        raise MetsiException("Missing 'freq_var' (e.g., 'stems_per_ha').")

    mode = operation_parameters.get("mode")
    if mode is None:
        raise MetsiException("Missing 'mode' (e.g., 'odds_units').")
    select_from_all = operation_parameters.get("select_from_all", False)
    if select_from_all is None:
        raise MetsiException("Missing 'select_from_all' (bool).")

    # Run selection
    removed_f = select_units(
        context=stand,
        data=trees,
        target_decl=target_decl,
        sets=py_sets,
        freq_var=freq_var,
        select_from_all=select_from_all,
        mode=mode,
    )

    # Strict checks
    if np.any(removed_f < 0):
        raise MetsiException("cutting produced negative removals; check tree_selection config.")
    over = removed_f > trees.stems_per_ha
    if np.any(over):
        raise MetsiException("cutting would remove more stems than available; fix targets/profiles.")

    # Apply removals
    if not trees.stems_per_ha.flags.writeable:
        trees.stems_per_ha = trees.stems_per_ha.copy()
    trees.stems_per_ha -= removed_f

    # Optional bookkeeping (only if explicitly provided)
    sim_time: Optional[int] = operation_parameters.get("sim_time")
    if sim_time is not None:
        stand.cutting_year = sim_time

    method = operation_parameters.get("cutting_method")
    if method is not None:
        stand.method_of_last_cutting = method


    return stand, []
