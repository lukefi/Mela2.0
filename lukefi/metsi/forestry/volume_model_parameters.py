import numpy as np
from numpy import typing as npt

from lukefi.metsi.app.utils import MetsiException

#                   Pine                Spruce              Birch
A_ALPHA =           (0.0136757779,      0.2370915100,       -0.6702075076   )
A_BETA =            (-0.0004330410,     -0.0022265039,      0.0007564219    )
A_GAMMA_FELLED =    (-0.0252589273,     0.0501848549,       -0.0208082140   )
A_GAMMA_SCANNED =   (0.0003471247,      0.0579317751,       -0.0493464118   )

B_ALPHA =           (0.0070401674,      -0.0242289073,      0.0091044957    )
B_BETA =            (-0.0001391022,     0,                  -0.0001411006   )
B_GAMMA_FELLED =    (-0.0058892178,     0,                  0               )
B_GAMMA_SCANNED =   (-0.0040737348,     0,                  0               )

C_ALPHA =           (-0.0110210343,     0.0065214188,       0.0308063633    )
C_BETA =            (0.0001268176,      0.0001064038,       0               )
C_GAMMA_FELLED =    (0.0090200383,      -0.0038986732,      0               )
C_GAMMA_SCANNED =   (0.0039640720,      -0.0051517384,      0               )

D =                 (-3.0483318645,     -3.6885804875,      -2.1571753518   )
E =                 (0,                 0.0001543815,       -0.0003581240   )
F =                 (1.0077373166,      2.9948108626,       2.7994786088    )

LAMBDA =            (-1.7357044890,     -1.3282974152,      -0.7831791889   )

SPECIES_PINES = [1, 8, 10, 12, 22, 29, 31]
SPECIES_SPRUCES = [2, 11, 14, 16, 19, 23, 32, 33, 34]
SPECIES_DECIDUOUS = [3, 4, 5, 6, 7, 9, 13, 15, 17, 18, 20, 21, 24, 25, 26, 27, 28, 30, 35, 36, 37, 38]


def volume_params(dbh: npt.NDArray[np.float64],
                  h: npt.NDArray[np.float64],
                  species: npt.NDArray[np.int32],
                  tempsum: float,
                  dataset: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate the logit transformed form factor ("logita") and lambda vectors for tree volume calculation

    :param dbh: breast height diameter
    :type dbh: npt.NDArray[np.float64]
    :param h: height
    :type h: npt.NDArray[np.float64]
    :param species: tree species
    :type species: npt.NDArray[np.int32]
    :param tempsum: temperature sum ("degree days")
    :type tempsum: float
    :param dataset: which dataset fit to use: "climbed", "felled" or "scanned"
    :type dataset: str
    :return: logita and lambda for volume calculation
    :rtype: tuple[NDArray[float64], NDArray[float64]]
    """
    species_in_pines = np.isin(species, SPECIES_PINES)
    species_in_spruces = np.isin(species, SPECIES_SPRUCES)
    species_in_deciduous = np.isin(species, SPECIES_DECIDUOUS)

    def get_param(param) -> npt.NDArray[np.float64]:
        return np.where(
            species_in_pines,
            param[0],
            np.where(
                species_in_spruces,
                param[1],
                np.where(
                    species_in_deciduous,
                    param[2],
                    0)))

    species_mapped = np.where(
        species_in_pines, 1, np.where(
            species_in_spruces, 2, np.where(
                species_in_deciduous, 3, 0)))

    if np.any(species_mapped == 0):
        raise MetsiException(f"Unknown species {list(species[species_mapped == 0])}")

    a_alpha = get_param(A_ALPHA)
    a_beta = get_param(A_BETA)
    b_alpha = get_param(B_ALPHA)
    b_beta = get_param(B_BETA)
    c_alpha = get_param(C_ALPHA)
    c_beta = get_param(C_BETA)

    a_gamma: npt.NDArray[np.float64] | float
    b_gamma: npt.NDArray[np.float64] | float
    c_gamma: npt.NDArray[np.float64] | float

    if dataset == "climbed":
        a_gamma = 0
        b_gamma = 0
        c_gamma = 0
    elif dataset == "felled":
        a_gamma = get_param(A_GAMMA_FELLED)
        b_gamma = get_param(B_GAMMA_FELLED)
        c_gamma = get_param(C_GAMMA_FELLED)
    elif dataset == "scanned":
        a_gamma = get_param(A_GAMMA_SCANNED)
        b_gamma = get_param(B_GAMMA_SCANNED)
        c_gamma = get_param(C_GAMMA_SCANNED)
    else:
        raise MetsiException(f"Unknown dataset type {dataset}")

    d = get_param(D)
    e = get_param(E)
    f = get_param(F)
    lambda_ = get_param(LAMBDA)

    logita = (a_alpha + a_beta * tempsum + a_gamma) + (b_alpha + b_beta * tempsum + b_gamma) * dbh + \
        (c_alpha + c_beta * tempsum + c_gamma) * h + d * 1 / h + e * dbh * h + f * 1 / (dbh * h)

    return (logita, lambda_)
