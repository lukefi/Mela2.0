from typing import Any, Optional
import numpy as np
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.data.util.select_units import select_units, SelectionSet, SelectionTarget


def _stand_metrics(trees: ReferenceTrees) -> dict[str, float]:
    stems = float(np.nansum(trees.stems_per_ha)) if trees.size > 0 else 0.0
    mean_dbh = float(np.nanmean(trees.breast_height_diameter)) if trees.size > 0 else float("nan")
    mean_h = float(np.nanmean(trees.height)) if trees.size > 0 else float("nan")
    return {"stems_per_ha_total": stems, "mean_dbh": mean_dbh, "mean_h": mean_h}


def _snapshot_removed(trees: ReferenceTrees, removed_f: np.ndarray) -> dict[str, Any]:
    sel = removed_f > 0
    idx = np.where(sel)[0]
    rows: list[dict[str, Any]] = []
    for i in idx:
        rows.append({
            "identifier": str(trees.identifier[i]),
            "species": int(trees.species[i]) if not np.isnan(trees.species[i]) else -1,
            "breast_height_diameter": float(trees.breast_height_diameter[i]),
            "height": float(trees.height[i]),
            "stems_per_ha_removed": float(removed_f[i]),
        })
    return {
        "count": int(idx.size),
        "removed_stems_per_ha_sum": float(np.nansum(removed_f[sel])) if idx.size else 0.0,
        "rows": rows,
    }


def cutting(input_: OpTuple[ForestStand], /, **operation_parameters) -> OpTuple[ForestStand]:
    """
    Generalized harvesting/thinning/ajourat treatment.

      - Requires params['tree_selection'] with 'Target' and 'sets'
      - Updates stand.cutting_year and stand.method_of_last_cutting

    Expected parameters:
      tree_selection: {
        Target: {type: str, var: str, amount: float},
        sets:   [ {sfunction, order_var, target_var, target_type, target_amount,
                   profile_x, profile_y, profile_xmode, (optional) profile_xscale}, ... ],
        (optional) mode: str
      }
      sim_time: int
      cutting_method: str 
      labels: list[str] (stored in cdata)

    Returns updated (stand, cdata).
    """
    stand, cdata = input_

    # --- guards
    if stand.reference_trees is None:
        raise MetsiException("cutting requires vectorized ReferenceTrees")

    # must have explicit tree_selection with Target + sets 
    ts: Optional[dict[str, Any]] = operation_parameters.get("tree_selection")
    if not ts or "Target" not in ts or "sets" not in ts:
        
        raise MetsiException("cutting requires 'tree_selection' with 'Target' and 'sets'.")

    target_dict: dict[str, Any] = ts["Target"]
    if not isinstance(target_dict, dict) or \
       ("type" not in target_dict or "var" not in target_dict or "amount" not in target_dict):
        raise MetsiException("tree_selection.Target must have keys: 'type', 'var', 'amount'.")

    target_decl = SelectionTarget()
    target_decl.type = target_dict["type"]
    target_decl.var = target_dict["var"]
    target_decl.amount = target_dict["amount"]

    sets_param = ts["sets"]
    if not isinstance(sets_param, (list, tuple)) or len(sets_param) == 0:
        raise MetsiException("tree_selection.sets must be a non-empty list of sets.")

    # Build sets exactly as given
    sets_py: list[SelectionSet[ForestStand, ReferenceTrees]] = []
    for i, s in enumerate(sets_param):
        try:
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
        except KeyError as e:
            raise MetsiException(f"Missing key in sets[{i}]: {e}") from e

        if ss.profile_x.shape != ss.profile_y.shape or ss.profile_x.ndim != 1 or ss.profile_x.size < 2:
            raise MetsiException(f"sets[{i}]: profile_x/profile_y must be 1D arrays of equal length (>=2).")
        sets_py.append(ss)

    # Optional things we still support
    mode: Optional[str] = ts.get("mode")  # R passes mode conditionally
    labels: Optional[list[str]] = operation_parameters.get("labels")
    sim_time: Optional[int] = operation_parameters.get("sim_time")

    # For select_units we keep freq_var explicit, but do not auto-map/guess.
    # In your data model 'stems_per_ha' corresponds to R's 'f'.
    # The sets' target_var should point to a valid column name in your SoA.
    freq_var: str = operation_parameters.get("freq_var", "stems_per_ha")

    # --- Run selection
    metrics_before = _stand_metrics(stand.reference_trees)

    removed_f = select_units(
        context=stand,
        data=stand.reference_trees,
        target_decl=target_decl,
        sets=sets_py,
        freq_var=freq_var,
        select_from_all=bool(operation_parameters.get("select_from_all", False)),
        mode=(mode if mode is not None else operation_parameters.get("mode", "odds_units")),
    )

    # STRICT validity: no negatives, no over-harvest
    if np.any(removed_f < 0):
        raise MetsiException("cutting produced negative removals; check target/profile configuration.")
    over_mask = removed_f > stand.reference_trees.stems_per_ha
    if np.any(over_mask):
        # crash rather than clamp
        idx = np.where(over_mask)[0][:5]  # show a few offenders
        raise MetsiException(
            f"cutting would remove more stems than available at {over_mask.sum()} units "
            f"(examples idx {idx.tolist()}); fix 'Target'/'sets'."
        )

    # Apply removals (no clamping)
    if not stand.reference_trees.stems_per_ha.flags.writeable:
        stand.reference_trees.stems_per_ha = stand.reference_trees.stems_per_ha.copy()
    stand.reference_trees.stems_per_ha -= removed_f

    metrics_after = _stand_metrics(stand.reference_trees)

    # --- Update stand fields per R (with robust fallback for key mismatch in R example)
    if sim_time is not None:
        stand.cutting_year=sim_time

    method = operation_parameters.get("cutting_method")
    if method is not None:
        stand.method_of_last_cutting = method


    # --- Collected data (kept from your richer Python impl)
    cdata.extend_list_result("removed_trees", [_snapshot_removed(stand.reference_trees, removed_f)])
    cdata.store("cutting", {
        "time": sim_time,
        "labels": labels or [],
        "freq_var": freq_var,
        "select_from_all": bool(operation_parameters.get("select_from_all", False)),
        "mode": (mode if mode is not None else operation_parameters.get("mode", "odds_units")),
        "target": {"type": target_decl.type, "var": target_decl.var, "amount": target_decl.amount},
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "removed_stems_per_ha": float(np.nansum(removed_f)),
    })

    return (stand, cdata)
