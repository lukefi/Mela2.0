import ctypes as cts

import numpy as np
from lukefi.metsi.data.enums.internal import Origin, SoilPeatlandCategory, DrainageCategory
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTree, ReferenceTrees

# DLL_PATH = Path('lukefi', 'metsi', 'forestry', 'fortran', 'lib', 'ikaf.dll')
DLL_PATH = '/dev/lib/ikaf.dll'
DLL = cts.CDLL(DLL_PATH)


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
    if tree.species <= 8:
        species = tree.species.value
    elif tree.species.is_coniferous():
        species = 7
    else:
        species = 8

    origin = 0 if tree.origin in (None, Origin.UNKNOWN, Origin.NATURAL_SEED, Origin.NATURAL_SPROUT) else 1

    if stand.drainage_category == DrainageCategory.TRANSFORMING_MIRE:
        drainage_category = 2
    elif stand.drainage_category == DrainageCategory.TRANSFORMED_MIRE:
        drainage_category = 3
    else:
        drainage_category = 1

    # intent(in), value
    rpl = cts.c_float(species)  # VMI7
    lpm = cts.c_float(tree.breast_height_diameter or 0.0)
    gy = cts.c_float(bal)
    lampos = cts.c_float(stand.degree_days or 0.0)
    kmy = cts.c_float((stand.geo_location[2] if stand.geo_location is not None else 0.0) or 0.0)
    boni = cts.c_float(stand.site_type_category or 0)  # VMI7 classification (same as internal)
    keskid = cts.c_float(dgm)
    rsynty = cts.c_float(origin)
    rverotar = cts.c_float(stand.tax_class_reduction or 0)  # VMI7 classification (same as in VMI12, VNI13)
    ojitil = cts.c_float(drainage_category)  # VMI7

    # intent(out)
    age_ptr = cts.c_float()
    hajo_ptr = cts.c_float()

    if stand.soil_peatland_category == SoilPeatlandCategory.MINERAL_SOIL:
        f = DLL.ages
        f.argtypes = [
            cts.c_float,
            cts.c_float,
            cts.c_float,
            cts.c_float,
            cts.c_float,
            cts.c_float,
            cts.c_float,
            cts.c_float,
            cts.c_float,
            cts.POINTER(cts.c_float),
            cts.POINTER(cts.c_float)
        ]
        f(rpl, lpm, gy, lampos, kmy, boni, keskid, rsynty, rverotar, cts.byref(age_ptr), cts.byref(hajo_ptr))

    else:  # peatlands
        f = DLL.agesuo
        f.argtypes = [
            cts.c_float,
            cts.c_float,
            cts.c_float,
            cts.c_float,
            cts.c_float,
            cts.c_float,
            cts.c_float,
            cts.c_float,
            cts.c_float,
            cts.c_float,
            cts.POINTER(cts.c_float),
            cts.POINTER(cts.c_float)
        ]
        f(rpl, lpm, gy, lampos, kmy, boni, keskid, rsynty, rverotar, ojitil, cts.byref(age_ptr), cts.byref(hajo_ptr))

    return age_ptr.value, age_ptr.value + added_years
