from enum import IntEnum

import numpy as np
from lukefi.metsi.data.enums.internal import Origin, SoilPeatlandCategory, DrainageCategory, TreeSpecies
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTree, ReferenceTrees
from lukefi.metsi.forestry.ika import ages as ages_
from lukefi.metsi.forestry.ika import agesuo as agesuo_


class _AgeModelSpecies(IntEnum):
    PINE = 1
    SPRUCE = 2
    SILVER_BIRCH = 3
    DOWNY_BIRCH = 4
    ASPEN = 5
    ALDER = 6
    OTHER_CONIFEROUS = 7
    OTHER_DECIDUOUS = 8


_SPECIES_MAP = {
    TreeSpecies.PINE: _AgeModelSpecies.PINE,
    TreeSpecies.SPRUCE: _AgeModelSpecies.SPRUCE,
    TreeSpecies.SILVER_BIRCH: _AgeModelSpecies.SILVER_BIRCH,
    TreeSpecies.DOWNY_BIRCH: _AgeModelSpecies.DOWNY_BIRCH,
    TreeSpecies.ASPEN: _AgeModelSpecies.ASPEN,
    TreeSpecies.GREY_ALDER: _AgeModelSpecies.ALDER,
    TreeSpecies.COMMON_ALDER: _AgeModelSpecies.ALDER,
    TreeSpecies.OTHER_CONIFEROUS: _AgeModelSpecies.OTHER_CONIFEROUS,
    TreeSpecies.OTHER_DECIDUOUS: _AgeModelSpecies.OTHER_DECIDUOUS
}


def ages(stand: ForestStand,
         trees_: ReferenceTrees,
         tree: ReferenceTree,
         added_years: float) -> tuple[float, float]:

    trees = trees_ if len(trees_) > 0 else stand.reference_trees

    # basal_area of larger trees
    large_mask = trees.breast_height_diameter > tree.breast_height_diameter
    bal = np.pi * np.sum(trees.stems_per_ha[large_mask] * ((trees.breast_height_diameter[large_mask] / 200) ** 2))
    bal = bal + 0.5 * tree.stems_per_ha * np.pi * ((tree.breast_height_diameter / 200) ** 2)

    # Dgm
    sd2 = np.sum(trees.stems_per_ha * trees.breast_height_diameter ** 2)
    sd3 = np.sum(trees.stems_per_ha * trees.breast_height_diameter ** 3)

    dgm = 0 if sd2 == 0 else sd3 / sd2

    # Model uses VMI7 classes
    if tree.species in _SPECIES_MAP:
        species = _SPECIES_MAP[tree.species]
    elif tree.species.is_coniferous():
        species = _AgeModelSpecies.OTHER_CONIFEROUS
    else:
        species = _AgeModelSpecies.OTHER_DECIDUOUS

    origin = 0 if tree.origin in (Origin.UNSET, Origin.UNKNOWN, Origin.NATURAL_SEED, Origin.NATURAL_SPROUT) else 1

    if stand.drainage_category == DrainageCategory.TRANSFORMING_MIRE:
        drainage_category = 2
    elif stand.drainage_category == DrainageCategory.TRANSFORMED_MIRE:
        drainage_category = 3
    else:
        drainage_category = 1

    # intent(in), value
    rpl = species
    lpm = tree.breast_height_diameter
    gy = bal
    lampos = stand.degree_days or 0.0
    kmy = (stand.geo_location[2] if stand.geo_location is not None else 0.0) or 0.0
    boni = min(stand.site_type_category or 0, 8)
    keskid = dgm
    rsynty = origin
    rverotar = stand.tax_class_reduction or 0
    ojitil = drainage_category

    if stand.soil_peatland_category == SoilPeatlandCategory.MINERAL_SOIL:
        age, _ = ages_(rpl, lpm, gy, lampos, kmy, boni, keskid, rsynty, rverotar)
    else:  # peatlands
        age, _ = agesuo_(rpl, lpm, gy, lampos, kmy, boni, keskid, rsynty, rverotar, ojitil)

    return age, age + added_years
