from typing import Callable, Optional
import numpy as np
import numpy.typing as npt

from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees


class SelectionSet:
    sfunction: Callable[[ForestStand, ReferenceTrees], npt.NDArray[np.bool_]]
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


def select_trees(stand: ForestStand,
                 trees: ReferenceTrees,
                 target_decl: SelectionTarget,
                 sets: list[SelectionSet],
                 freq_var: str = "stems_per_ha",
                 select_from: str = "all",
                 mode: str = "odds_trees"):

    if target_decl.var is not None and target_decl.type is not None and target_decl.amount is not None:
        target_var = target_decl.var
        target_type = target_decl.type
        target_amount = target_decl.amount
    else:
        target_var = None
        target_type = None
        target_amount = None

    # tarkista, että y-arvot välillä [0,1]
    for set_ in sets:
        if (np.max(set_.profile_y) > 1) or (np.min(set_.profile_y) < 0):
            raise MetsiException("Invalid y value in profile. Should be between 0 and 1.")

    # kerättävä määrä
    all_trees: npt.NDArray[np.bool_] = np.repeat(True, trees.size)
    total_target = _get_target(trees, all_trees, target_var, target_type, target_amount)

    # sallittu toleranssi --> hyväksyttävä väli [target-eps, target+ eps]
    # epsilon
    tolerance = 0.001
    eps_total = max(0.005, min(total_target * tolerance, 100))
    eps_step = 0.0001  # lopetetaan binäärihaku, jos askel tätä pienempi

    # valitut runkoluvut
    selected_stems: npt.NDArray[np.float64] = np.repeat(0.0, trees.size)
    tmp_stems: npt.NDArray[np.float64] = np.repeat(0.0, trees.size)
    stems: npt.NDArray[np.float64] = np.repeat(0.0, trees.size)
    total_target_selected: float = 0.0
    tmp_total_target: float = 0.0
    cur_set_target_var = ""
    tmp_cur_set_target: float = 0.0

    # valintajoukot, käydään läpi vain kokonaistavoitteeseen tarvittavat joukot
    i_set = 0
    while not ((total_target - eps_total) <= total_target_selected <= (total_target + eps_total)) and i_set < len(sets):
        # joukkoon kuuluvat puut
        cur_set_mask = sets[i_set].sfunction(stand, trees)
        if np.any(cur_set_mask):
            cur_set_idx = np.nonzero(cur_set_mask)[0]
            # cur_trees = trees[cur_set_idx]
            # järjestysmja ja 'kerättävä' muuttuja
            cur_set_order_var = sets[i_set].order_var
            cur_set_target_var = sets[i_set].target_var
            # joukon puut järjestettynä
            cur_set_idx_ord = np.argsort(trees[cur_set_order_var][cur_set_idx])
            # cur_trees_ordered = cur_trees[cur_set_idx_ord]
            # joukon tavoitemäärä
            cur_set_target = _get_target(trees,
                                         cur_set_mask,
                                         sets[i_set].target_var,
                                         sets[i_set].target_type,
                                         sets[i_set].target_amount)
            eps_set = max(0.005, cur_set_target * tolerance)
            # lasketaan alkuarvo vakiolle tai skaalaukselle profiilin ensimmäiselllä janalla
            # muiden janojen vakiot riippuvat tästä
            # Alkuarvosta eteenpäin binäärihaulla
            if sets[i_set].profile_xmode == "relative":
                if hasattr(sets[i_set], "profile_xscale") and sets[i_set].profile_xscale == "all":
                    ord_x_min = np.min(trees[cur_set_order_var])
                    ord_x_max = np.max(trees[cur_set_order_var])
                else:
                    ord_x_min = trees[cur_set_order_var][cur_set_idx_ord][0]
                    ord_x_max = trees[cur_set_order_var][cur_set_idx_ord][-1]
                d_ord = ord_x_max - ord_x_min
                # käytetään sisäisesti aina absoluuttista x-asteikkoa
                d_profile_x = sets[i_set].profile_x[-1] - sets[i_set].profile_x[0]
                prof_x = ord_x_min + (sets[i_set].profile_x - sets[i_set].profile_x[0]) * d_ord / d_profile_x
            else:
                # absolute#
                prof_x = sets[i_set].profile_x
            prof_y = sets[i_set].profile_y

            tmp_total_target = total_target_selected
            tmp_cur_set_target = 0.0
            tmp_stems = selected_stems.copy()

            # lasketaan kullekin joukon puulle alkuosuus y_i (skaalaus 0-1)
            i_ordx = 0  # puun järjestysindeksi
            y: npt.NDArray[np.float64] = np.repeat(0.0, trees.size)
            # lm: kulmakertoimet ja vakiotermit vektoreissa
            b: npt.NDArray[np.float64] = np.diff(prof_y) / np.diff(prof_x)
            # Jos vain yksi order_var-mjan arvo ja suhteellinen x, niin prof_x:n kaikki pisteet ovat samoja
            # skaalataan siinä tapauksesa vain vakio-osaa a (kulmakertoimet b nolliksi)
            if np.any(np.isnan(b)):
                b = np.repeat(0, b.size)

            a: npt.NDArray[np.float64] = prof_y[1:] - b * prof_x[1:]
            bounds_x: npt.NDArray[np.float64] = np.insert([-np.inf, np.inf], [1], prof_x[1:-1])
            interval_func = np.vectorize(lambda x: np.arange(b.size)[(x >= bounds_x[:-1]) & (x < bounds_x[1:])])
            interval_id: npt.NDArray[np.integer] = interval_func(trees[cur_set_order_var][cur_set_idx_ord])

            # puukohtainen osuus lähtötilanteessa
            for i_ordx in range(cur_set_mask.size):
                idx = cur_set_idx[cur_set_idx_ord[i_ordx]]  # järjestysindeksiä vastaava puun alkuperäinen indeksi
                y[idx] = max(0.0, min(1.0, a[interval_id[i_ordx]] + b[interval_id[i_ordx]]
                             * trees[cur_set_order_var][cur_set_idx_ord][i_ordx]))
            # tässä joukossa valittu runkoluku, ei voi ylittää jäljellä olevaa määrää
            if select_from == "all":
                stems = np.minimum(y * trees[freq_var], trees[freq_var] - selected_stems)
            else:
                stems = np.maximum(0.0, y * (trees[freq_var] - selected_stems))
            if target_var is None or target_var == freq_var:
                target = np.sum(stems)
            else:
                target = np.sum(stems * trees[target_var])
            if cur_set_target_var is None or cur_set_target_var == freq_var:
                sub_target = np.sum(stems)
            else:
                sub_target = np.sum(stems * trees[cur_set_target_var])
            tmp_total_target = tmp_total_target + target
            tmp_cur_set_target = tmp_cur_set_target + sub_target
            tmp_stems = selected_stems + stems

            # jos valintajoukon tavoite annettu, etsitään lopullinen profiili
            # binäärihaulla, muuten käytetään suoraan annettua profiilia

            if not np.isinf(cur_set_target):
                # initial values for binary search
                scale, step, y0 = _init_search(mode,
                                               y,
                                               prof_y,
                                               total_target,
                                               tmp_total_target,
                                               cur_set_target,
                                               tmp_cur_set_target)
                # binäärihaku
                # jatketaan kunnes riittävän lähellä tavoitetta (kokonais- tai
                # valintajoukon) tai kaikki valintajoukon puut valittu
                while ((tmp_cur_set_target < (cur_set_target - eps_set) and
                        tmp_total_target < (total_target - eps_total) and
                        np.sum(trees[freq_var][cur_set_idx]) > np.sum(tmp_stems[cur_set_idx])) or
                       ((cur_set_target + eps_set) < tmp_cur_set_target or
                        (total_target + eps_total) < tmp_total_target)) and step > eps_step:

                    tmp_total_target = total_target_selected
                    tmp_cur_set_target = 0

                    # uusi osuuskandidaatti
                    y = _scale_y(mode,
                                 y0,
                                 scale,
                                 prof_x,
                                 prof_y,
                                 interval_id,
                                 trees,
                                 cur_set_idx_ord,
                                 cur_set_order_var)

                    if np.any(np.isnan(y)):
                        raise MetsiException("Unable to continue binary search")

                    if select_from == "all":
                        stems = np.minimum(y * trees[freq_var], trees[freq_var] - selected_stems)
                    else:
                        stems = np.maximum(0.0, y * (trees[freq_var] - selected_stems))

                    if target_var is None or target_var == freq_var:
                        target = np.sum(stems)
                    else:
                        target = np.sum(stems * trees[target_var])

                    if cur_set_target_var is None or cur_set_target_var == freq_var:
                        sub_target = np.sum(stems)
                    else:
                        sub_target = np.sum(stems * trees[cur_set_target_var])

                    tmp_total_target = tmp_total_target + target
                    tmp_cur_set_target = tmp_cur_set_target + sub_target
                    tmp_stems = selected_stems + stems

                    # seuraavan kierroksen skaalaus/vakio
                    step = step / 2
                    if tmp_cur_set_target > (cur_set_target + eps_set) or tmp_total_target > (total_target + eps_total):
                        scale = scale - step
                    else:
                        scale = scale + step
                # while searching
            # if not set has target
        # if not empty set

        selected_stems = tmp_stems.copy()
        total_target_selected = tmp_total_target

        print(f"{i_set}, {cur_set_target_var}, {tmp_cur_set_target}, {np.sum(stems)}")
        i_set = i_set + 1
    # sets

    print(f"{target_var}, {total_target_selected}")
    return selected_stems


def _odds(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return p / (1 - p)


def _i_odds(o: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    val = o / (1 + o)
    val[np.isinf(o)] = 1
    return val


def _get_target(trees: ReferenceTrees,
                current_set: npt.NDArray[np.bool_],
                target_var: Optional[str],
                target_type: Optional[str],
                target_amount: Optional[float],
                freq_var: str = "stems_per_ha"):

    if target_var is None or target_type is None or target_amount is None:
        return np.inf

    if target_type == "absolute":
        amount = target_amount

    elif target_type == "relative":
        if target_var == freq_var:
            amount = target_amount * np.sum(trees[freq_var][current_set])
        else:
            amount = target_amount * np.sum(trees[freq_var][current_set] * trees[target_var][current_set])

    elif target_type == "absolute_remain":
        if target_var == freq_var:
            amount = np.sum(trees[freq_var][current_set]) - target_amount
        else:
            amount = np.sum(trees[freq_var][current_set] * trees[target_var][current_set]) - target_amount

    elif target_type == "relative_remain":
        if target_var == freq_var:
            amount = (1 - target_amount) * np.sum(trees[freq_var][current_set])
        else:
            amount = (1 - target_amount) * np.sum(trees[freq_var][current_set] * trees[target_var][current_set])

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

    # Ääriarvot binäärihaulle

    if mode in ("odds_profile", "odds_trees"):
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
             trees: ReferenceTrees,
             cur_set_idx_ord: npt.NDArray[np.integer],
             cur_set_order_var: str):

    odds_y0 = y0.copy()
    y = y0.copy()

    if mode == "odds_profile":
        prof_y = _i_odds(scale * odds_y0)

        # Uudet kulmakertoimet ja kuvauspuiden osuudet
        b = np.diff(prof_y) / np.diff(prof_x)
        # Jos vain yksi ja sama order_var-mjan arvo ja suhteellinen x, niin prof_x:n kaikki pisteet ovat samoja
        # skaalataan siinä tapauksessa vain vakio-osaa a (kulmakertoimet b nolliksi)
        if np.any(np.isnan(b)):
            b = np.repeat(0, b.size)

        a = prof_y[1:] - b * prof_x[1:]
        for i_ordx in range(trees.size):
            idx = cur_set_idx_ord[i_ordx]
            y[idx] = max(0.0, min(1.0, a[interval_id[i_ordx]] + b[interval_id[i_ordx]]
                                  * trees[cur_set_order_var][cur_set_idx_ord][i_ordx]))

    elif mode == "odds_trees":
        y[cur_set_idx_ord] = _i_odds(scale * odds_y0[cur_set_idx_ord])

    elif mode == "scale":
        y[cur_set_idx_ord] = np.maximum(0.0, np.minimum(1.0, scale * y0[cur_set_idx_ord]))

    elif mode == "level":
        y[cur_set_idx_ord] = np.maximum(0.0, np.minimum(1.0, y0[cur_set_idx_ord] + scale))

    return y
