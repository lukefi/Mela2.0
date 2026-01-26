import ctypes as cts
from pathlib import Path
import math
from lukefi.metsi.data.enums.internal import SoilPeatlandCategory, DrainageCategory, Origin
from lukefi.metsi.data.model import ForestStand, ReferenceTree

#DLL_PATH = Path('lukefi', 'metsi', 'forestry', 'fortran', 'lib', 'ikaf.dll')
DLL_PATH = '/dev/lib/ikaf.dll'
DLL = cts.CDLL(DLL_PATH)

def ages(stand: ForestStand, tree: ReferenceTree, added_years: float, trees_:list[ReferenceTree]=[]) -> tuple[float, float]:

    trees = trees_ if len(trees_)>0 else stand.reference_trees
    
    #basal_area of larger trees
    large = [t for t in trees if t.breast_height_diameter > tree.breast_height_diameter]
    bal = sum([t.stems_per_ha * math.pi*((t.breast_height_diameter/200)**2) for t in large])
    bal = bal + 0.5*tree.stems_per_ha * math.pi*((tree.breast_height_diameter/200)**2)
    
    #Dgm
    Sd2 = sum([t.stems_per_ha * t.breast_height_diameter**2 for t in trees])
    Sd3 = sum([t.stems_per_ha * t.breast_height_diameter**3 for t in trees])
    dgm = 0 if Sd2 == 0 else Sd3/Sd2
    
    #Model uses VMI7 classes
    if tree.species <= 8:
        species = tree.species
    elif tree.species.is_coniferous:
        species = 7
    else:
        species = 8
        
    origin = 0 if tree.origin in (None, Origin.UNKNOWN,Origin.NATURAL_SEED,Origin.NATURAL_SPROUT) else 1
    
    if stand.drainage_category == DrainageCategory.TRANSFORMING_MIRE:
        drainage_category = 2
    elif stand.drainage_category == DrainageCategory.TRANSFORMED_MIRE:
        drainage_category = 3
    else:
        drainage_category = 1
    
    #intent(in), value
    rpl = cts.c_float(species) #VMI7
    lpm = cts.c_float(tree.breast_height_diameter or 0.0)
    gy = cts.c_float(bal)
    lampos = cts.c_float(stand.degree_days)
    kmy = cts.c_float(stand.geo_location[2])
    boni = cts.c_float(stand.site_type_category) #VMI7 classification (same as internal)
    #keskid = cts.c_float(Sd3/Sd2)
    keskid = cts.c_float(dgm)
    rsynty = cts.c_float(origin)
    rverotar = cts.c_float(stand.tax_class_reduction or 0) #VMI7 classification (same as in VMI12, VNI13)
    ojitil = cts.c_float(drainage_category) #VMI7
    
    #intent(out)
    age_ptr = cts.c_float()
    hajo_ptr = cts.c_float()
    
    if stand.soil_peatland_category == SoilPeatlandCategory.MINERAL_SOIL: #
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
        f(rpl, lpm, gy, lampos, kmy, boni, keskid, rsynty,rverotar,cts.byref(age_ptr), cts.byref(hajo_ptr))
    
    else: #peatlands
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
        f(rpl, lpm, gy, lampos, kmy, boni, keskid, rsynty,rverotar,ojitil,cts.byref(age_ptr), cts.byref(hajo_ptr))
    
    return age_ptr.value, age_ptr.value + added_years