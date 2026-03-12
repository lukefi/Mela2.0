from rpy2 import robjects
from lukefi.metsi.data.enums.vmi import VmiIteration
from lukefi.metsi.forestry.preprocessing.tree_generation_lm import lm_tree_generation_loaded
from lukefi.metsi.forestry.r.hdmod.hdmod_files import HDMOD_FILE_PATH_MAP


def predict_tree_height(
        nfi_iteration: VmiIteration,
        model: int,
        temp_sum: float,
        stratum_weighted_mean_diameter: float,
        tree_diameter: float,
        stratum_basal_area: float) -> float:
    global lm_tree_generation_loaded

    if not lm_tree_generation_loaded:
        robjects.r.source("lukefi/metsi/forestry/r/lm_tree_generation.R")
        lm_tree_generation_loaded = True

    hdmod_path = HDMOD_FILE_PATH_MAP[nfi_iteration]

    r_args = {
        "HDpath": hdmod_path,
        "whichmodel": model,
        "dd": temp_sum,
        "DGM": stratum_weighted_mean_diameter or robjects.NA_Real,
        "dbh": tree_diameter,
        "Gos": stratum_basal_area
    }

    result = robjects.r['Hpred_simple'](**r_args)

    return result
