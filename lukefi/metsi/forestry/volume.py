from enum import StrEnum
import numpy as np
from numpy import typing as npt

from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.forestry.volume_model_parameters import volume_params


class TreeVolumeDataset(StrEnum):
    CLIMBED = "climbed"
    FELLED = "felled"
    SCANNED = "scanned"


def tree_volumes(reference_trees: ReferenceTrees,
                 temperature_sum: float,
                 dataset: TreeVolumeDataset = TreeVolumeDataset.CLIMBED):
    dbh = reference_trees.breast_height_diameter
    h = reference_trees.height
    species = reference_trees.species
    logita, lambda_ = volume_params(dbh, h, species, temperature_sum, dataset)

    return _tree_volumes(dbh, h, logita, lambda_)


def _tree_volumes(breast_height_diameter: npt.NDArray[np.float64],
                  height: npt.NDArray[np.float64],
                  logita: npt.NDArray[np.float64],
                  lambda_: npt.NDArray[np.float64]):
    lam = np.exp(lambda_)
    w = 2 - 2 * np.exp((height - 1.3) / lam) / (1 + np.exp((height - 1.3) / lam))
    rstump = w * breast_height_diameter / 20 + (1 - w) * breast_height_diameter / 20 * height / (height - 1.3)
    return np.pi * np.exp(logita) / (1 + np.exp(logita)) * (rstump) ** 2 * (10 * height)
