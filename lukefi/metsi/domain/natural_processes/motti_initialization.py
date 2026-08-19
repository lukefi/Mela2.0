from typing import Any

import numpy as np
from lukefi.metsi.data.conversion import internal2motti
from lukefi.metsi.data.enums.internal import CuttingMethod, Storey, TreeSpecies
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.motti.motti_types import MottiState
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.domain.natural_processes import motti_util
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL


# NOTE: Miksi tämän osalta ei käytetä sitä mitä on enumeissa?
FDM_TO_MOTTI_STOREY = {
    Storey.DOMINANT: 2,  # ylempi
    Storey.UNDER: 1,     # alempi
    Storey.OVER: 3,      # siemenpuu
    Storey.SPARE: 4,     # säästöpuu
}


def _storey_to_motti(
    stand: ForestStand,
    index: int,
    fdm_storey: Storey,
    *,
    is_stratum_index: bool = False,
) -> int:
    """
    Convert FDM Storey -> Motti puustojakso/puuluokka.

    Exact classes:
      DOMINANT -> 2
      UNDER    -> 1
      OVER     -> 3
      SPARE    -> 4

    Fallback:
      - if only one stratum: ylempi=2
      - if multiple strata and this stratum is clearly lower:
          height gap > 5m and lower stratum height < 10m -> alempi=1
      - otherwise ylempi=2

    Parameters
    ----------
    index:
        If is_stratum_index=True, this is a direct tree_strata row index.
        Otherwise it is assumed to be a reference_trees row index, and the
        matching stratum row is resolved through rt.stratum -> strata.stratum_number.
    """
    if fdm_storey in FDM_TO_MOTTI_STOREY:  # NOTE: FDM_TO_MOTTI_STOREY vois vaihtaa interl2motti.py
        return FDM_TO_MOTTI_STOREY[fdm_storey]

    strata = stand.tree_strata
    if strata is None or strata.size <= 1:
        return 2

    stratum_idx: int | None = None

    if is_stratum_index:
        if 0 <= index < strata.size:
            stratum_idx = index
    else:
        rt = stand.reference_trees
        if 0 <= index < rt.size:
            target_sid = int(rt.stratum[index])
            if target_sid > 0:
                for j in range(strata.size):
                    sid = int(strata.stratum_number[j])
                    if sid == target_sid:
                        stratum_idx = j
                        break

    if stratum_idx is None:
        return 2

    heights = np.nan_to_num(strata.mean_height, nan=0.0)
    current_h = float(heights[stratum_idx])
    max_h = float(np.max(heights))

    if (max_h - current_h) > 5.0 and current_h < 10.0:
        return 1

    return 2


def _strip_tree_strata(stand: ForestStand):
    """
    Clear tree information from strata
    """
    if stand.tree_strata.size == 0:
        return

    n = stand.tree_strata.size
    stripped = TreeStrata(size=n)

    stripped.identifier = stand.tree_strata.identifier.copy()
    stripped.origin = stand.tree_strata.origin.copy()
    stripped.storey = stand.tree_strata.storey.copy()

    stripped.basal_area[:] = 0.0
    stripped.stems_per_ha[:] = 0.0
    stripped.mean_height[:] = 0.0
    stripped.mean_diameter[:] = 0.0
    stripped.breast_height_age[:] = 0.0
    stripped.biological_age[:] = 0.0
    stripped.sapling_stems_per_ha[:] = 0.0
    stripped.number_of_generated_trees[:] = 0

    stand.tree_strata = stripped

# kutsujärjestys 1. _init_motti_state


def _spedom(rt: ReferenceTrees) -> int:
    """
    NOTE: pitäskö tää laskea uudestaan joka stepissä, jos puulajijakauma muuttuu merkittävästi?
     - Vaikuttaako Mottiin jos ei päivitetä?
    NOTE: onko tämä vähän epäluotettava, koska riippuu siitä että reference_trees on kunnossa ja 
        että siellä on lajikoodit.
    - Ehkä vois olla parempi hakea suoraan Mottista dominantti laji ja laskea spedom siitä?
        - NOTE: laskeeko mottiInit itse spedom vai onko tää hyvä antaa tätä kautta?
        - Returns dominant species from Motti species.
    NOTE: onko tämän oikea paikka tälle vai pitäskö olla conversion/interl2motti.py:ssä? Tai jopa motti_util.py:ssä?

    Prefer basal area totals; if BA totals are all zero/missing, fall back to stems/ha.
    If trees are empty fall back to PINE, we need to give valid value for growth.
    """
    if rt.size == 0:
        return TreeSpecies.PINE

    # Convert species to Motti codes (will raise if invalid)
    spe_codes = [internal2motti.convert_species(TreeSpecies(int(s))) for s in rt.species]

    # Basal area per tree: stems_per_ha * π * (0.5 * d_cm * 0.01 m/cm)^2
    d_cm = np.nan_to_num(rt.breast_height_diameter, nan=0.0)
    f_ha = np.nan_to_num(rt.stems_per_ha, nan=0.0)
    ba_per_tree = f_ha * np.pi * (0.5 * d_cm * 0.01) ** 2  # m²/ha contribution

    # Sum BA per species code
    ba_per_species: dict[int, float] = {}
    for code, ba in zip(spe_codes, ba_per_tree.tolist()):
        ba_per_species[code] = ba_per_species.get(code, 0.0) + float(ba)

    use_basal = any(v > 0.0 for v in ba_per_species.values())
    if not use_basal:
        ba_per_species.clear()
        # Fallback: stems/ha totals per species
        for code, stems in zip(spe_codes, f_ha.tolist()):
            # TODO: Is this correct?
            ba_per_species[code] = ba_per_species.get(code, 0.0) + float(stems)

    if not ba_per_species:
        return TreeSpecies.PINE

    return max(ba_per_species.items(), key=lambda kv: kv[1])[0]


def _auto_euref_km(geo_location:
                   Optional[tuple[float | None,
                                  float | None,
                                  float | None,
                                  CRS | None]]) -> tuple[float, float]:
    """ Normalize to EUREF-FIN/TM35FIN kilometers. """
    if geo_location is None:
        raise ValueError("Stand is missing geolocation information required by Motti")

    x, y, _, crs = geo_location

    if crs is None or crs not in CRS.EPSG_3067:
        raise ValueError("Expected EUREF-FIN/TM35 in kilometers.")
    if not x or not y:
        raise ValueError("Stand is missing coordinates values")
    
    return x / 1000.0, y / 1000.0


def _build_motti_strata_py(stand: ForestStand, strata: TreeStrata | None = None) -> list[dict[str, float]]:
    """
    Convert given TreeStrata into Python dicts for Motti4Strata.
    If strata is not given, use stand.tree_strata.

    Uncertain fields:
      hw -> temporary fallback to mean_height
      dg -> temporary fallback to mean_diameter
      st -> temporary dummy 0.0
    """
    if strata is None:
        strata = stand.tree_strata

    if strata.size == 0:
        return []

    out: list[dict[str, float]] = []

    for i in range(min(strata.size, 10)):
        species = TreeSpecies(int(strata.species[i]))
        if species <= TreeSpecies.TREELESS:  # TODO: parempi olisi käyttää enumeja. + species.TREELESS
            continue

        biological_age = float(np.nan_to_num(strata.biological_age[i], nan=0.0))
        basal_area = float(np.nan_to_num(strata.basal_area[i], nan=0.0))  # TODO: lisää ehto - jos ppa < 0.001 --> 0.0
        stems_main = float(np.nan_to_num(strata.stems_per_ha[i], nan=0.0))
        mean_height = float(np.nan_to_num(strata.mean_height[i], nan=0.0))
        mean_diameter = float(np.nan_to_num(strata.mean_diameter[i], nan=0.0))
        origin = float(strata.origin[i])

        storey = _storey_to_motti(
            stand,
            i,
            Storey(int(strata.storey[i])),
            is_stratum_index=True,
        )

        stratum_sid = int(strata.stratum_number[i])
        if stratum_sid <= 0:
            stratum_sid = i + 1

        spe = float(internal2motti.convert_species(species))
        out.append({
            "spe": spe,
            "age": biological_age,
            "ba": basal_area,
            "f": stems_main,
            # nää on aritmeettisia ei keskiarvoja. TODO: Mitä tehdään - tyhjänä, nollana vai lasketaan (kuka laskee)?
            "h": mean_height,
            "hw": mean_height,
            "d": mean_diameter,  # nää on aritmeettisia ei keskiarvoja.
            "dg": mean_diameter,
            "storey": storey,
            "st": origin,
            "sid": float(stratum_sid),
        })

    return out


def _compress_strata_for_motti(strata: TreeStrata, max_strata: int = 10) -> TreeStrata:
    """
    If there are more than max_strata strata, merge säästöpuut into one so the count becomes max_strata.

    Candidate Säästöpuu for merge is:
      - number_of_generated_trees == 1
      - storey == SPARE

    Merged result:
      - species = species whose candidate strata have the highest total stems_per_ha
      - mean_height = avg
      - mean_diameter = avg
      - stems_per_ha = sum
      - storey / origin / stratum_rank / stratum_number / identifier = from base row

    If there are not enough merge candidates, return original strata unchanged.
    """
    if strata.size <= max_strata:
        return strata

    excess = strata.size - max_strata
    if excess <= 0:
        return strata

    # Iso kuva:
    # KPG_lukupuille(luku_puu) -> säästöpuu_osite, jolla (tunnistusehto) n_gen=1 ja storey=SPARE *1
    # - compress siis tiivistää vain KPG:n generoimia säästöpuu
    # - alkuperäisiin ositteisiin ei kosketa, KPG:n generoima säästöpuu "ylivuoto" on tarkoitus tiivistää. ***
    # säästöpuu
    # ongelmadomain: motti ei tue kuin 10 ositetta max., joten ylimääräiset ei mahdu mukaan.
    # - säästöpuu spesifi UC (kuvaupuiden generoinnissa tehdään säästöpuut)
    # Tässä vaiheessa ositteissa on datassa luetut (oikeat) ositteet + KPG luomat säästöpuuositteet
    candidate_idx: list[int] = []  # tähän KPG:n generoimat säästöpuu ositteet
    for i in range(strata.size):
        # KPG on lukenut lukupuun ja muodostanut siitä säästöpuuositteen. Siksi n_gen == 1 *1
        n_gen = int(np.nan_to_num(strata.number_of_generated_trees[i], nan=0))
        storey = int(np.nan_to_num(strata.storey[i], nan=-1))
        if n_gen == 1 and storey == int(Storey.SPARE):
            candidate_idx.append(i)

    needed = excess + 1
    if len(candidate_idx) < needed:
        return strata  # fallback: current truncation behavior stays

    # take exactly as many as needed; simplest and least invasive
    merge_idx = candidate_idx[:needed]

    # species totals by stems_per_ha -> choose dominant/base species
    stems_by_species: dict[int, float] = {}
    for i in merge_idx:
        species = int(strata.species[i])
        stems = float(np.nan_to_num(strata.stems_per_ha[i], nan=0.0))
        stems_by_species[species] = stems_by_species.get(species, 0.0) + stems

    base_species = max(stems_by_species.items(), key=lambda kv: kv[1])[0]

    # choose base row as first row of the major species
    base_idx = next(i for i in merge_idx if int(strata.species[i]) == base_species)
    rest_idx = [i for i in merge_idx if i != base_idx]  # TODO: rest_idx osittelle pitää vaihtaa base_idx osite_id

    out = strata[:]

    # merged numeric values
    out.stems_per_ha[base_idx] = float(np.nansum(out.stems_per_ha[merge_idx]))
    out.mean_height[base_idx] = float(np.nanmean(out.mean_height[merge_idx]))
    out.mean_diameter[base_idx] = float(np.nanmean(out.mean_diameter[merge_idx]))

    if np.any(~np.isnan(out.biological_age[merge_idx])):
        out.biological_age[base_idx] = float(np.nanmean(out.biological_age[merge_idx]))

    if np.any(~np.isnan(out.breast_height_age[merge_idx])):
        out.breast_height_age[base_idx] = float(np.nanmean(out.breast_height_age[merge_idx]))

    out.sapling_stems_per_ha[base_idx] = float(np.nansum(out.sapling_stems_per_ha[merge_idx]))

    # force species to same
    out.species[base_idx] = base_species

    if rest_idx:
        out.delete(rest_idx)

    return out


def _init_motti_state(stand: ForestStand) -> MottiState:
    """Initialize and attach persistent MottiState to stand if missing."""

    spedom = _spedom(stand.reference_trees)

    y_km, x_km = _auto_euref_km(stand.geo_location)

    if stand.geo_location is not None:
        z = stand.geo_location[2]
        if z is None or z == 0.0:
            z = -1.0
    else:
        z = -1.0

    yy = Motti4DLL.new_site(
        Y=y_km,
        X=x_km,
        Z=z,
        lake=stand.lake_effect if stand.lake_effect is not None else 0.0,
        sea=stand.sea_effect if stand.sea_effect is not None else 0.0,
        mal=stand.land_use_category.value if stand.land_use_category is not None else 0,
        mty=internal2motti.resolve_site_type(
            stand.drained_peatland_type,
            stand.site_type_category),
        verl=stand.tax_class if stand.tax_class is not None else 0,
        verlt=stand.tax_class_reduction if stand.tax_class_reduction is not None else 0,
        xt_regen=((stand.year - stand.artificial_regeneration_year)
                  if stand.artificial_regeneration_year is not None
                  else -9999),
        xt_muok=((stand.year - stand.soil_surface_preparation_year)
                 if stand.soil_surface_preparation_year is not None
                 else -9999),
        xt_raiv=((stand.year - stand.regeneration_area_cleaning_year)
                 if stand.regeneration_area_cleaning_year is not None
                 else -9999),
        sid=stand.stand_id or 0,
        fthin=stand.method_of_last_cutting in (CuttingMethod.THINNING, CuttingMethod.FIRST_THINNING),
        xt_thin=((stand.year - stand.cutting_year)
                    if stand.cutting_year is not None and
                       stand.method_of_last_cutting not in (CuttingMethod.CLEARCUTTING, CuttingMethod.NO_CUTTING)
                    else -9999),
        xt_fert=((stand.year - stand.fertilization_year)
                 if stand.fertilization_year is not None
                 else -9999),
        xt_thoit=((stand.year - stand.young_stand_tending_year)
                  if stand.young_stand_tending_year is not None
                  else -9999),
        drain=internal2motti.convert_drainage_category(stand.drainage_category),
        xt_ndrain=((stand.year - stand.drainage_year)
                   if stand.drainage_year is not None
                   else -9999),
        alr=stand.soil_peatland_category.value if stand.soil_peatland_category is not None else 0,
        year=stand.year - stand.start_year,
        spedom=spedom,  # OK
        spedom2=spedom,  # OK pääpuulajimetsikkö
        nstorey=1.0,
        gstorey=1.0,
    )

    rt = stand.reference_trees
    n = rt.size
    # TODO: Is this right? yp <--> rt yhteys, tää pitää tarkastella onko
    # validi näin; eg. menetetäänkö alkuperäiset ja miten strata päivitys
    # vaikuttaa.
    rt.tree_number = np.arange(1, n + 1, dtype=rt.tree_number.dtype)
    ids = rt.tree_number.astype(int).copy()

    stems = np.nan_to_num(rt.stems_per_ha, nan=0.0)
    d13 = np.nan_to_num(rt.breast_height_diameter, nan=0.0)
    h = np.nan_to_num(rt.height, nan=0.0)
    age = np.nan_to_num(rt.biological_age, nan=0.0)
    age13 = np.nan_to_num(rt.breast_height_age, nan=0.0)

    # TODO: ReferenceTrees does not have this attribute; where did it come from?
    cr = np.nan_to_num(getattr(rt, "crown_ratio", np.zeros(n, dtype=float)), nan=0.0)
    origin = rt.origin
    spe_vec = [internal2motti.convert_species(TreeSpecies(int(s))) for s in rt.species]

    stratum_ids = [
        # Miksi? else osassa otetaan kuviolta id? eikö stratum_id:t mene sekaisin?
        int(v) if v > 0 else (stand.stand_id or (idx + 1))
        for idx, v in enumerate(rt.stratum)
    ]
    storey_vec = [_storey_to_motti(stand, idx, Storey(int(rt.storey[idx]))) for idx in range(n)]
    trees_py = [
        {
            "id": int(i),
            "sid": int(sid),
            "f": float(f),
            "d13": float(d),
            "h": float(hh),
            "spe": int(sp),
            "age": float(a),
            "age13": float(a13),
            "cr": float(c),
            "snt": int(o + 1),
            "storie": float(storey),

        }
        for i, sid, f, d, hh, sp, a, a13, c, o, storey in zip(
            ids,
            stratum_ids,  # original osite_id
            stems,
            d13,
            h,
            spe_vec,
            age,
            age13,
            cr,
            origin,
            storey_vec,
        )
    ]
    yp, ntrees = Motti4DLL.new_trees(trees_py)

    # puut on jo yp:ssä ja niiden osite id:t muuttu. eli hävikääkö puut joiden pitäisi siirtyä yp vektoriin?
    # _compress* == jos liikaa ositteita, niin tiivistetään s.e. jää vain 10 ositetta.
    #   - ylimääräisistä siirretään puut yhteen ositteeseen eg. 5 ositetta --> jäljelle jäävään
    #
    # Tää pitäs tehdä ennen yp:n alustusta. Rikkoo osite_id eheyden.
    compressed_strata = _compress_strata_for_motti(stand.tree_strata, max_strata=10)
    strata_py = _build_motti_strata_py(stand, compressed_strata)

    yo = Motti4DLL.new_strata(strata_py)

    buffers = Motti4DLL.alloc_state_buffers()
    buffers.ctrl.death_tree = 1

    ntrees = Motti4DLL.initialize_with_state(
        yo=yo,
        yy=yy,
        yp=yp,
        numtrees=ntrees,
        buffers=buffers,
    )

    _strip_tree_strata(stand)

    return MottiState(yy=yy, yp=yp, ntrees=ntrees, buffers=buffers, )


def initialize_motti(stand: ForestStand, **_: dict[str, Any]) -> None:
    """ Initialize MottiState for forest stand if missing. Does nothing if already initialized. """
    if stand.motti_state is None:
        stand.motti_state = _init_motti_state(stand)
        motti_util.reconcile_reference_trees_from_motti(stand, init_mode=True)


__all__ = ["initialize_motti"]
