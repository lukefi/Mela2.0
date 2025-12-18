from enum import StrEnum
import numpy as np
from numpy import typing as npt

from lukefi.metsi.data.vector_model import ReferenceTrees
from lukefi.metsi.forestry.volume_model_parameters import volume_params


class TreeVolumeDataset(StrEnum):
    CLIMBED = "climbed"
    """
    Data from 1968-1972 from the plots of the fifth NFI collected by climbing living trees and performing
    measurements at multiple heights.
    """

    FELLED = "felled"
    """
    Data from 1988-2001. The trees were felled and measured on the ground.
    """

    SCANNED = "scanned"
    """
    Data measured by terrestrial laser scanning. Plots selected from the 12th NFI.
    """


def tree_volumes(reference_trees: ReferenceTrees,
                 temperature_sum: float,
                 dataset: TreeVolumeDataset = TreeVolumeDataset.CLIMBED) -> npt.NDArray[np.float64]:
    """
    Calculate volumes for reference trees based on variable form factor model.

    :param reference_trees: reference trees whose volumes to calculate
    :type reference_trees: ReferenceTrees
    :param temperature_sum: temperature sum ("degree days")
    :type temperature_sum: float
    :param dataset: which dataset fit to use for model parameters ("climbed", "felled" or "scanned")
    :type dataset: TreeVolumeDataset
    :return: vector containing calculated volumes for each reference tree
    :rtype: npt.NDArray[np.float64]
    """
    dbh = reference_trees.breast_height_diameter
    h = reference_trees.height
    species = reference_trees.species
    logita, lambda_ = volume_params(dbh, h, species, temperature_sum / 10, dataset)

    return _tree_volumes(dbh, h, logita, lambda_) / 1000


def _tree_volumes(dbh: npt.NDArray[np.float64],
                  h: npt.NDArray[np.float64],
                  logita: npt.NDArray[np.float64],
                  lambda_: npt.NDArray[np.float64]):
    lam = np.exp(lambda_)
    w = 2 - 2 * np.exp((h - 1.3) / lam) / (1 + np.exp((h - 1.3) / lam))
    rstump = w * dbh / 20 + (1 - w) * dbh / 20 * h / (h - 1.3)
    return np.pi * np.exp(logita) / (1 + np.exp(logita)) * (rstump) ** 2 * (10 * h)
