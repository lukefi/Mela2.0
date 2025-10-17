from typing import Callable, Optional
import numpy as np
import numpy.typing as npt

from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.vector_model import VectorData


class SelectionSet[T, V: VectorData]:
    sfunction: Callable[[T, V], npt.NDArray[np.bool_]]
    order_var: str
    target_var: str
    target_type: str
    target_amount: float
    profile_x: npt.NDArray[np.float64]
    profile_y: npt.NDArray[np.float64]
    profile_xmode: str
    profile_xscale: Optional[str]


class SelectionTarget:
    type: str
    var: str
    amount: float


def select_units[T, V: VectorData](context: T,
                                   data: V,
                                   target_decl: SelectionTarget,
                                   sets: list[SelectionSet[T, V]],
                                   freq_var: str,
                                   select_from: str = "all",
                                   mode: str = "odds_units") -> npt.NDArray[np.float64]:
    """
    Summary:
        The tree selection routine returns for each reference tree the number of the stems (per ha) needed to meet the
        required amount of a given tree-level variable.
        The selection routine is controlled by:
        1. The overall target which is described by the required amount of the given tree-level variable and the type
            of the target.
            Given a target amount A and target variable V, the four target types are
            - "absolute": selected stems accumulate A units of variable V
            - "relative": selected stems accumulate A percentages of variable V, 0 <= A <= 1
            - "absolute_remain": selected stems accumulate Vtot-A units of variable V, Vtot = sum of all reference trees
            - "relative_remain": selected stems accumulate 1-A percentages of variable V, 0 <= A <= 1
        2.  1-N selection sets, each specifying the subset of trees eligible for selection, target for the set and a
            profile. From the profile routine interpolates the proportion to be selected from the stems of each tree.
        
        The subset of trees eligible for selection is defined by a function that should return true for those trees
        belonging to the selection set and false for those trees that cannot be selected.
        
        The target of the set is defined the same way as the overall target. No more stems are selected from the set
        than are required to achieve the set's target. The sets are deployed in the given order and only as many sets
        are deployed as are needed for the overall target.
        
        The profile consists of continuous linear segments represented by x and y coordinates. The x coordinates
        correspond to the given tree level variable used to sort the trees. The y coordinates represent the relative
        amounts at each segment's end points. x coordinates can be either absolute (corresponding to actual values of
        the order variable) or relative (highest/lowest x coordinate corresponds to highest/lowest value of the order
        variable in the set).
        
        For each segment a linear equation y = a + b * x is generated. The tree selection routine adjusts the linear
        equations to achieve the overall and sets targets. The adjustment is done using binary search and there are four
        adjustment modes:
        - "level" : binary search is used to find a constant C that is added to all segments i.e. y = a + b * x + C.
          The basic idea is to retain shape of the profile and lift/lower it. As y values are restricted to be in range
          [0, 1] the shape may change as y reach it's limits.
        - "scale" : binary search is used to find a constant C that is a multiplier to all segments i.e.
          y = (a + b*x) * C. The basic idea is to retain the ratio of y values. As y values are restricted to be in
          range [0, 1] the ratio may change as y reach it's limits.
        - "odds_profile" : binary search is used to find a constant C that is a multiplier of the odds of end points of
          segments.  i.e y = odds_inv(C*odds(a+b*x)), where odds_inv is the inverse of odds(y)=y/1-y. The basic idea is
          to have a flexible profile shape so that the closer profile values are 0 and 1 the less the profile values are
          changed. The new y values are calculated for the end points of the segments so for each iteration and new
          value of C the new linear equations are generated to calculate new proportions of stems for each tree.
        - "odds_tree" (default): like odds_profile, but the y values to be scaled are the proportions for each
          individual reference tree instead of the end points  of segments.
        
        As a result of the binary search the values y are the proportions of stems to be selected. If option select_from
        is set to "all", the number of selected stems is computed always as a proportion of the initial number of stems
        of each tree. Otherwise number of selected stems is computed as a proportion of what is left after selections
        from previous selection sets.

    Args:
        context:        Free-form contextual information
        data:           Where the units are selected from
        target_decl:    Description of selection target
        sets:           List of SelectionSets describing the different kinds of selection methods to use in order to
                        reach the total target
        freq_var:       The name of the variable to describe the frequency of units in the data
        select_from:
        mode:           Selection mode to use

    Returns:
        npt.NDArray[np.float64]: The selected units for each element in data
    """
    if target_decl.var is not None and target_decl.type is not None and target_decl.amount is not None:
        target_var = target_decl.var
        target_type = target_decl.type
        target_amount = target_decl.amount
    else:
        target_var = None
        target_type = None
        target_amount = None

    # check that y values are between [0,1]
    for set_ in sets:
        if (np.max(set_.profile_y) > 1) or (np.min(set_.profile_y) < 0):
            raise MetsiException("Invalid y value in profile. Should be between 0 and 1.")

    # amount to collect
    all_units: npt.NDArray[np.bool_] = np.repeat(True, data.size)
    total_target = _get_target(data, all_units, target_var, target_type, target_amount, freq_var)

    # allowed tolerance --> acceptable interval [target - eps, target + eps]
    # epsilon
    tolerance = 0.001
    eps_total = max(0.005, min(total_target * tolerance, 100))
    eps_step = 0.0001  # stop binary search if step is smaller than this

    # chosen variables
    selected_units: npt.NDArray[np.float64] = np.repeat(0.0, data.size)
    tmp_units: npt.NDArray[np.float64] = np.repeat(0.0, data.size)
    units: npt.NDArray[np.float64] = np.repeat(0.0, data.size)
    total_target_selected: float = 0.0
    tmp_total_target: float = 0.0
    cur_set_target_var = ""
    tmp_cur_set_target: float = 0.0

    # selection sets, only iterate through sets needed for total target
    i_set = 0
    while not ((total_target - eps_total) <= total_target_selected <= (total_target + eps_total)) and i_set < len(sets):
        # rows in set
        cur_set_mask = sets[i_set].sfunction(context, data)
        if np.any(cur_set_mask):
            cur_set_idx = np.nonzero(cur_set_mask)[0]

            # order variable and 'collected' variable
            cur_set_order_var = sets[i_set].order_var
            cur_set_target_var = sets[i_set].target_var

            # rows in the set ordered
            cur_set_idx_ord = np.argsort(data[cur_set_order_var][cur_set_idx])

            # target for the set
            cur_set_target = _get_target(data,
                                         cur_set_mask,
                                         sets[i_set].target_var,
                                         sets[i_set].target_type,
                                         sets[i_set].target_amount,
                                         freq_var)
            eps_set = max(0.005, cur_set_target * tolerance)

            # calculate initial value for constant or scaling on the first line segment of the profile
            # constants for other line segments depend on this
            # Continue with binary search starting from the initial value
            if sets[i_set].profile_xmode == "relative":
                if hasattr(sets[i_set], "profile_xscale") and sets[i_set].profile_xscale == "all":
                    ord_x_min = np.min(data[cur_set_order_var])
                    ord_x_max = np.max(data[cur_set_order_var])
                else:
                    ord_x_min = data[cur_set_order_var][cur_set_idx][cur_set_idx_ord][0]
                    ord_x_max = data[cur_set_order_var][cur_set_idx][cur_set_idx_ord][-1]

                d_ord = ord_x_max - ord_x_min

                # always use absolute x scale internally
                d_profile_x = sets[i_set].profile_x[-1] - sets[i_set].profile_x[0]
                prof_x = ord_x_min + (sets[i_set].profile_x - sets[i_set].profile_x[0]) * d_ord / d_profile_x
            else:
                # absolute#
                prof_x = sets[i_set].profile_x

            prof_y = sets[i_set].profile_y

            tmp_total_target = total_target_selected
            tmp_cur_set_target = 0.0
            tmp_units = selected_units.copy()

            # calculate initial share y_i for each row of the set (scaling 0-1)
            i_ordx = 0  # rivin järjestysindeksi
            y: npt.NDArray[np.float64] = np.repeat(0.0, data.size)

            # lm: slopes and constants in vectors
            b: npt.NDArray[np.float64] = np.diff(prof_y) / np.diff(prof_x)

            # if only one order_var value and relative x, all points in prof_x are the same
            # in that case scale only constant part a (set slopes b to zero)
            if np.any(np.isnan(b)):
                b = np.repeat(0, b.size)

            a: npt.NDArray[np.float64] = prof_y[1:] - b * prof_x[1:]
            bounds_x: npt.NDArray[np.float64] = np.insert([-np.inf, np.inf], [1], prof_x[1:-1])
            interval_func = np.vectorize(lambda x: np.arange(b.size)[(x >= bounds_x[:-1]) & (x < bounds_x[1:])])
            interval_id: npt.NDArray[np.integer] = interval_func(data[cur_set_order_var][cur_set_idx][cur_set_idx_ord])

            # shares at starting per row
            for i_ordx in range(cur_set_idx_ord.size):
                idx = cur_set_idx[cur_set_idx_ord[i_ordx]]  # original row index corresponding to order index
                y[idx] = max(0.0, min(1.0, a[interval_id[i_ordx]] + b[interval_id[i_ordx]]
                             * data[cur_set_order_var][cur_set_idx][cur_set_idx_ord][i_ordx]))

            # chosen variable in this set, can not exceed remaining amount
            if select_from == "all":
                units = np.minimum(y * data[freq_var], data[freq_var] - selected_units)
            else:
                units = np.maximum(0.0, y * (data[freq_var] - selected_units))

            # special case: select all units from set
            if (target_type == "relative") and (sets[i_set].target_type == "relative") and (target_amount == 1) and \
                    (sets[i_set].target_amount == 1):
                units = np.repeat(0.0, data.size)
                units[cur_set_idx] = data[freq_var][cur_set_idx]

            if target_var is None or target_var == freq_var:
                target = np.sum(units)
            else:
                target = np.sum(units * data[target_var])

            if cur_set_target_var is None or cur_set_target_var == freq_var:
                sub_target = np.sum(units)
            else:
                sub_target = np.sum(units * data[cur_set_target_var])

            tmp_total_target = tmp_total_target + target
            tmp_cur_set_target = tmp_cur_set_target + sub_target
            tmp_units = selected_units + units

            # if selection set target is given, search for final profile with binary search
            # otherwise use given profile directly

            if not np.isinf(cur_set_target):
                # initial values for binary search
                scale, step, y0 = _init_search(mode,
                                               y,
                                               prof_y,
                                               total_target,
                                               tmp_total_target,
                                               cur_set_target,
                                               tmp_cur_set_target)

                # binary search
                # continue until close enough to target (total or selection set)
                # or until all units in the selection set have been selected
                while ((tmp_cur_set_target < (cur_set_target - eps_set) and
                        tmp_total_target < (total_target - eps_total) and
                        np.sum(data[freq_var][cur_set_idx]) > np.sum(tmp_units[cur_set_idx])) or
                       ((cur_set_target + eps_set) < tmp_cur_set_target or
                        (total_target + eps_total) < tmp_total_target)) and step > eps_step:

                    tmp_total_target = total_target_selected
                    tmp_cur_set_target = 0

                    # new share candidate
                    y = _scale_y(mode,
                                 y0,
                                 scale,
                                 prof_x,
                                 prof_y,
                                 interval_id,
                                 data,
                                 cur_set_idx[cur_set_idx_ord],
                                 cur_set_order_var)

                    if np.any(np.isnan(y)):
                        raise MetsiException("Unable to continue binary search")

                    if select_from == "all":
                        units = np.minimum(y * data[freq_var], data[freq_var] - selected_units)
                    else:
                        units = np.maximum(0.0, y * (data[freq_var] - selected_units))

                    if target_var is None or target_var == freq_var:
                        target = np.sum(units)
                    else:
                        target = np.sum(units * data[target_var])

                    if cur_set_target_var is None or cur_set_target_var == freq_var:
                        sub_target = np.sum(units)
                    else:
                        sub_target = np.sum(units * data[cur_set_target_var])

                    tmp_total_target = tmp_total_target + target
                    tmp_cur_set_target = tmp_cur_set_target + sub_target
                    tmp_units = selected_units + units

                    # scaling/constant for next round
                    step = step / 2
                    if tmp_cur_set_target > (cur_set_target + eps_set) or tmp_total_target > (total_target + eps_total):
                        scale = scale - step
                    else:
                        scale = scale + step

                # while searching
            # if not set has target
        # if not empty set

        selected_units = tmp_units.copy()
        total_target_selected = tmp_total_target

        i_set = i_set + 1
    # sets

    return selected_units


def _odds(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return p / (1 - p)


def _i_odds(o: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    val = o / (1 + o)
    val[np.isinf(o)] = 1
    return val


def _get_target(data: VectorData,
                current_set: npt.NDArray[np.bool_],
                target_var: Optional[str],
                target_type: Optional[str],
                target_amount: Optional[float],
                freq_var: str) -> float:

    if target_var is None or target_type is None or target_amount is None:
        return np.inf

    if target_type == "absolute":
        amount = target_amount

    elif target_type == "relative":
        if target_var == freq_var:
            amount = target_amount * np.sum(data[freq_var][current_set])
        else:
            amount = target_amount * np.sum(data[freq_var][current_set] * data[target_var][current_set])

    elif target_type == "absolute_remain":
        if target_var == freq_var:
            amount = np.sum(data[freq_var][current_set]) - target_amount
        else:
            amount = np.sum(data[freq_var][current_set] * data[target_var][current_set]) - target_amount

    elif target_type == "relative_remain":
        if target_var == freq_var:
            amount = (1 - target_amount) * np.sum(data[freq_var][current_set])
        else:
            amount = (1 - target_amount) * np.sum(data[freq_var][current_set] * data[target_var][current_set])

    else:
        raise MetsiException(f"Unknown target type {target_type}")

    return max(0.0, amount)


def _init_search(mode: str,
                 y: npt.NDArray[np.float64],
                 prof_y: npt.NDArray[np.float64],
                 total_target: float,
                 tmp_total_target: float,
                 cur_set_target: float,
                 tmp_cur_set_target: float) -> tuple[float, float, npt.NDArray[np.float64]]:

    # Limits for binary search

    if mode in ("odds_profile", "odds_units"):
        if mode == "odds_profile":
            odds_y0 = _odds(prof_y)
        else:
            odds_y0 = _odds(y)

        y0 = odds_y0

        if tmp_total_target > total_target or tmp_cur_set_target > cur_set_target:
            scale = 0.500001
            step = 1.0
        else:
            scale = 501.0
            step = 1000.0

    elif mode == "scale":
        scalemax = np.max(1 / y[y > 0])
        if np.any(y == 0):
            scalemax = np.maximum(scalemax, 100.0)
        scale = scalemax / 2
        step = scalemax / 2
        y0 = y

    elif mode == "level":
        a_max = np.max(1 - y)
        a_min = np.min(-1 * y)
        scale = (a_max + a_min) / 2
        step = abs(scale)
        if step == 0:
            step = a_max
        y0 = y

    else:
        raise MetsiException(f"Unknown mode {mode}")

    return scale, step, y0


def _scale_y(mode: str,
             y0: npt.NDArray[np.float64],
             scale: float,
             prof_x: npt.NDArray[np.float64],
             prof_y: npt.NDArray[np.float64],
             interval_id: npt.NDArray[np.integer],
             data: VectorData,
             cur_set_idx_ord: npt.NDArray[np.integer],
             cur_set_order_var: str) -> npt.NDArray[np.float64]:

    odds_y0 = y0.copy()
    y = y0.copy()

    if mode == "odds_profile":
        prof_y = _i_odds(scale * odds_y0)
        y = np.empty((data.size), dtype=np.float64)

        # New slopes and row shares
        b = np.diff(prof_y) / np.diff(prof_x)

        # if only one order_var value and relative x, all points in prof_x are the same
        # in that case scale only constant part a (set slopes b to zero)
        if np.any(np.isnan(b)):
            b = np.repeat(0, b.size)

        a = prof_y[1:] - b * prof_x[1:]

        for i_ordx in range(cur_set_idx_ord.size):
            idx = cur_set_idx_ord[i_ordx]
            y[idx] = max(0.0, min(1.0, a[interval_id[i_ordx]] + b[interval_id[i_ordx]]
                                  * data[cur_set_order_var][cur_set_idx_ord][i_ordx]))

    elif mode == "odds_units":
        y[cur_set_idx_ord] = _i_odds(scale * odds_y0[cur_set_idx_ord])

    elif mode == "scale":
        y[cur_set_idx_ord] = np.maximum(0.0, np.minimum(1.0, scale * y0[cur_set_idx_ord]))

    elif mode == "level":
        y[cur_set_idx_ord] = np.maximum(0.0, np.minimum(1.0, y0[cur_set_idx_ord] + scale))

    return y
