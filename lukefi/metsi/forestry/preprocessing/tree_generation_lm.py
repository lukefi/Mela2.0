from pathlib import Path
import math
from typing import Any
from rpy2 import robjects

from lukefi.metsi.data.enums.internal import TreeSpecies
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStratum
from lukefi.metsi.forestry.preprocessing.pljak import get_spe_proportions
from lukefi.metsi.forestry.preprocessing.dhkertoimet import DHCOEFF
from lukefi.metsi.forestry.preprocessing.dgmean_kitumaa import DGMEAN_KITUMAA

SPECIES_INT2LM = [
    # Mänty 1, Kuusi 2, Rkoivu 3, Hkoivu 4,Haapa 5,Hleppä 6, Tleppä 7, Muu
    # havupuu 20, Muu lehtipuu 30, Douglaskuusi -> Muu kuusi 16,
    1, 2, 3, 4, 5, 6, 7, 20, 30, 16,
    # 11: Kataja 18, Kontorta 11, Kynäjalava 22, Lehtikuusi 14, Metsälehmus
    # 24,  Mustakuusi -> Muu kuusi 16, Paju -> Raita 9, Pihlaja 8, Pihta 15,
    # Raita 9
    18, 11, 22, 14, 24, 16, 9, 8, 15, 9,
    # 21: Saarni 26, Setri -> Muu mänty 13, Serbian kuusi -> Muu kuusi 16,
    # Tammi 27, Tuomi 28, Vaahtera 29, Visakoivu -> Muu lehtipuu 30,
    # Vuorijalava 23, Muu havupuu 20, Muu lehtipuu 30
    26, 13, 16, 27, 28, 29, 30, 23, 20, 30,
    # 31: Muu mänty 13, 32: Muu kuusi 16, Tuija 17, Marjakuusi 19, Halava 21,
    # Poppeli 25, Pähkinäpensas 31, Tuntematon -> Muu lehtipuu
    13, 16, 17, 19, 21, 25, 31, 30
]

SPECIES_LM2INT: list[TreeSpecies] = [
    # Mänty, kuusi, koivu, hkoivu, haapa, hleppä, tleppä, pihlaja, raita, puuton
    TreeSpecies(1), TreeSpecies(2), TreeSpecies(3), TreeSpecies(4), TreeSpecies(5), TreeSpecies(6), TreeSpecies(7),
    TreeSpecies(18), TreeSpecies(20), TreeSpecies.TREELESS,
    # kontorta, sembra, muu mänty, lehtikuusi, pihta, muu kuusi, tuija, kataja, marjakuusi, muu havupuu
    TreeSpecies(12), TreeSpecies(22), TreeSpecies(31), TreeSpecies(14), TreeSpecies(19), TreeSpecies(32),
    TreeSpecies(33), TreeSpecies(11), TreeSpecies(34), TreeSpecies(8),
    # halava, kynäjalava, vuorijalava, metsälehmus, poppeli, saarni, tammi, tuomi, vaahtera,muu lehtipuu, pähkinä
    TreeSpecies(35), TreeSpecies(13), TreeSpecies(28), TreeSpecies(15), TreeSpecies(36), TreeSpecies(21),
    TreeSpecies(24), TreeSpecies(25), TreeSpecies(26), TreeSpecies(9), TreeSpecies(37)
]

lm_tree_generation_loaded = False  # pylint: disable=invalid-name


def _determine_hmalli_value(species: TreeSpecies):
    if species in (TreeSpecies.PINE, TreeSpecies.OTHER_PINE, TreeSpecies.SHORE_PINE):
        return 1
    if species.is_coniferous():
        return 2
    return 3


def tree_generation_lm(stand: ForestStand, stratum: TreeStratum, **params) -> ReferenceTrees:
    global lm_tree_generation_loaded  # pylint: disable=global-statement
    dir_ = Path(__file__).parent.parent.resolve() / "r"
    growth_script_file = dir_ / "lm_tree_generation.R"
    if not lm_tree_generation_loaded:
        robjects.r.source(str(growth_script_file))
        lm_tree_generation_loaded = True

    degree_days = stand.degree_days
    stand_basal_area = stand.basal_area
    stand_land_use_cat = stand.land_use_category
    stand_county = stand.region
    stand_municipality = stand.municipality_id
    stand_development_class = stand.development_class
    if stand_municipality in (47, 148, 890):
        stand_county = 30

    nos = stratum.stems_per_ha
    gos = stratum.basal_area

    spevmi = SPECIES_INT2LM[stratum.species.value - 1]
    # kitumaalle keskiläpimitta taulukosta
    if stand_land_use_cat == 2:
        dgmean: dict[str, Any] = next(
            (item for item in DGMEAN_KITUMAA if item["maakunta"] == stand_county and item["species"] == spevmi), {
                "maakunta": 0, "species": 0, "DGM": 0.0})
        dgm: float = dgmean["DGM"]
    else:
        dgm = stratum.mean_diameter

    stratum_data = {
        'DGM': robjects.FloatVector([dgm]),
        'HGM': robjects.FloatVector([stratum.mean_height]),
        'G': robjects.FloatVector([stand_basal_area]),
        'Gos': robjects.FloatVector([gos]),
        'spe': robjects.FloatVector([stratum.species.value]),
        'DDY': robjects.FloatVector([degree_days]),
        'Nos': robjects.FloatVector([nos or robjects.NA_Real])
    }

    dhcoeffs = next(
        (item for item in DHCOEFF
         if item["maakunta"] == stand_county and
         item["maalk"] == stand_land_use_cat and
         item["puulaji"] == spevmi),
        {"maakunta": 0, "maalk": 0, "puulaji": 0, "dfactor": None, "hfactor": None}
    )
    dhcoeffs_vec = robjects.FloatVector([dhcoeffs["dfactor"], dhcoeffs["hfactor"]]) if dhcoeffs["dfactor"] is not None \
        else None

    assert stand_land_use_cat is not None
    assert stand_county is not None
    assert stand_development_class is not None

    species_proportions = get_spe_proportions(stand_land_use_cat, stand_county, stand_development_class,
                                              stratum.asema, stratum.mean_diameter, stratum.stems_per_ha, spevmi)
    proportions_data = {
        'puulaji': robjects.FloatVector(list(range(1, len(species_proportions) + 1))),
        'osuus': robjects.FloatVector(species_proportions)
    }

    # source_trees = stratum.__dict__.get('_trees', [])
    source_trees: ReferenceTrees = stand.reference_trees[stand.reference_trees.stratum == stratum.identifier]

    tree_data = {
        # 'lpm': robjects.FloatVector([tree.breast_height_diameter or robjects.NA_Real for tree in source_trees]),
        'lpm': robjects.FloatVector([source_trees.breast_height_diameter[i] for i in range(len(source_trees))]),
        'height': robjects.FloatVector([robjects.NA_Real
                                        if source_trees.tuhon_ilmiasu[i] in ('2', '61', '62', '71', '72') or
                                        source_trees.measured_height[i] == 0
                                        else (source_trees.measured_height[i]) for i in range(len(source_trees))]),
        'lkm': robjects.FloatVector([source_trees.stems_per_ha[i] or
                                     robjects.NA_Real for i in range(len(source_trees))])
    }

    gos_div = params.get('lm_gos_div', 1)
    # pyydetty max kuvauspuiden lkm
    ntrees_max = params.get('lm_n_trees', params.get('n_trees', 10))

    # ppa:han perustuva kuvauspuiden lkm
    ntrees = 2 if stratum.basal_area is None else min(max(math.floor(stratum.basal_area / gos_div), 2), ntrees_max)
    # jos kuvauspuiden lkm tyyppi on "param", käytetään annettua, ei ppasta laskettua määrää
    ntrees = ntrees_max if params.get('lm_n_trees_mode', 'param') == 'param' else ntrees

    # plosia käytettäessä kuvauspuiden lkm voi räjähtää, siksi skaalataan plosin summalla
    if sum(species_proportions) > 0:
        ntrees = ntrees / sum(species_proportions)

    df = robjects.DataFrame(stratum_data)
    df2 = robjects.DataFrame(tree_data)
    dfplos = robjects.DataFrame(proportions_data) if stand_land_use_cat == 1 else None

    r_args_all = {
        'ositerivi': df,
        'lukupuut': df2,
        'path': str(dir_) + '/',
        'tapa': params.get('lm_mode', 'dcons'),
        'width': params.get('lm_fix_width', 2),
        'n': ntrees,
        'hmalli': _determine_hmalli_value(stratum.species),
        'shdef': params.get('lm_shdef', 5),
        'shinit': 0.1,
        'plos': dfplos,
        'nmax': params.get('lm_stems_nmax', 2000),
        'dhfactor': dhcoeffs_vec
    }

    r_args = {k: v for k, v in r_args_all.items() if v is not None}
    result_df = robjects.r['generoi.kuvauspuut'](**r_args)

    index_stems = 12 if params.get('stems_mode', 'lkm') == 'lkm' else 11
    retval = ReferenceTrees(result_df.nrow)

    for i in range(result_df.nrow):
        treespe = SPECIES_LM2INT[int(result_df.rx2(9)[i]) - 1] if stand_land_use_cat == 1 else result_df.rx2(9)[i]
        retval.breast_height_diameter[i] = result_df.rx2(10)[i]
        retval.stems_per_ha[i] = result_df.rx2(index_stems)[i]
        retval.height[i] = result_df.rx2(13)[i]
        retval.species[i] = -1 if result_df.rx2(9)[i] == robjects.NA_Integer else treespe
        retval.biological_age[i] = stratum.biological_age
        retval.sapling[i] = result_df.rx2(13)[i] < 1.3

    return retval
