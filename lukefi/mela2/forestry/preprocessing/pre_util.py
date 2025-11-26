""" Module contains basic, domain and state spesific utility functions used in preprocessing operations"""
from lukefi.mela2.data.model import TreeStratum


# ---- state utils ----


def supplement_mean_diameter(stratum: TreeStratum) -> TreeStratum:
    diameter_factor = 1.2
    stratum.mean_diameter = (stratum.mean_height * diameter_factor) if stratum.mean_height is not None else None
    return stratum
