from typing import Any

import numpy as np
import numpy.typing as npt

from lukefi.metsi.data.conversion.internal2motti import convert_species
from lukefi.metsi.data.enums.internal import Origin, Storey, TreeSpecies
from lukefi.metsi.data.model import ForestStand, MottiState
from lukefi.metsi.data.motti.motti_types import Motti4SaplingStratum
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.domain.natural_processes.util import new_reference_tree_identity
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL

UT_CATEGORIES = [
    ("kkp", "usable"),
    ("klv", "unusable"),
    ("vlj", "farmed"),
]

UT_SPECIES_FIELDS = [
    ("ma", TreeSpecies.PINE),
    ("ku", TreeSpecies.SPRUCE),
    ("ra", TreeSpecies.SILVER_BIRCH),
    ("hi", TreeSpecies.DOWNY_BIRCH),
    ("ha", TreeSpecies.ASPEN),
    ("hl", TreeSpecies.GREY_ALDER),
    ("tl", TreeSpecies.OTHER_DECIDUOUS),
    ("mh", TreeSpecies.OTHER_CONIFEROUS),
    ("ml", TreeSpecies.OTHER_DECIDUOUS),
    ("_10", TreeSpecies.UNSET),
]

FDM_TO_MOTTI_STOREY = {
    Storey.DOMINANT: 2,  # ylempi
    Storey.UNDER: 1,     # alempi
    Storey.OVER: 3,      # siemenpuu
    Storey.SPARE: 4,     # säästöpuu
}


def auto_euref_km(y1: float | None, x1: float | None) -> tuple[float, float]:
    """
    Normalize to EUREF-FIN/TM35FIN kilometers.
    Input is expected to be in meters
    - Raise if values look like lat/long.
    """
    if not y1 or not x1:
        raise ValueError("Stand is missing coordinates required by Motti")
    abs_y, abs_x = abs(y1), abs(x1)

    # Clear lat/long guard
    if abs_y <= 90.0 and abs_x <= 180.0:
        raise ValueError(
            f"Coordinates look like lat/long (Y={y1}, X={x1}). "
            "Expected EUREF-FIN/TM35 in kilometers."
        )

    return y1 / 1000.0, x1 / 1000.0


def safe_origin(raw: float | int) -> int:
    v = int(raw)
    return v if v >= 0 else int(Origin.UNSET)


def next_osite_id(stand: ForestStand) -> int:
    used: list[int] = []

    rt = stand.reference_trees
    if rt.size > 0:
        for v in rt.stratum:
            x = int(v)
            if x > 0:
                used.append(int(v))

    ms = stand.motti_state
    if ms is not None and ms.buffers is not None:
        ut = ms.buffers.saplings
        for layer in range(10):
            for spe_name, _ in UT_SPECIES_FIELDS:
                s = getattr(ut[0][layer], spe_name)
                for cat_code, _ in UT_CATEGORIES:
                    x = int(getattr(s, f"osid_{cat_code}", 0))
                    if x > 0:
                        used.append(x)

    if ms is not None and ms.yp is not None:
        for i in range(int(ms.ntrees or 0)):
            x = int(ms.yp[0][i].sid)
            if x > 0:
                used.append(x)

    return (max(used) + 1) if used else 1


def reference_tree_indices_by_stratum(rt: ReferenceTrees, osid: int) -> list[int]:
    target = str(int(osid))
    retval: list[int] = []
    for i, value in enumerate(rt.stratum.tolist()):
        if str(value) == target:
            retval.append(i)
    return retval


def find_non_sapling_reference_tree_index(rt: ReferenceTrees, osid: int, tree_number: int) -> int | None:
    target_tree_number = int(tree_number)
    for i in reference_tree_indices_by_stratum(rt, osid):
        if bool(rt.sapling[i]):
            continue
        try:
            if int(rt.tree_number[i]) == target_tree_number:
                return i
        except (TypeError, ValueError):
            continue
    return None


def storey_from_layer(stand: ForestStand, layer: int) -> int:
    strata = getattr(stand, "tree_strata", None)
    if strata is None or layer >= strata.size:
        return int(Storey.UNSET)

    try:
        v = int(strata.storey[layer])
        return v if v >= 0 else int(Storey.UNSET)
    except (TypeError, ValueError):
        return int(Storey.UNSET)


def find_sapling_reference_tree_index(rt: ReferenceTrees, osid: int) -> int | None:
    target_osid = int(osid)

    for i in reference_tree_indices_by_stratum(rt, osid):
        if not bool(rt.sapling[i]):
            continue
        try:
            if int(rt.stratum[i]) == target_osid:
                return i
        except (TypeError, ValueError):
            continue

    return None


def storey_to_motti(
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
    if fdm_storey in FDM_TO_MOTTI_STOREY:
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


def sync_yp_to_reference_trees(stand: ForestStand) -> None:
    ms = stand.motti_state
    if ms is None or ms.yp is None:
        return

    yp = ms.yp
    rt = stand.reference_trees

    for i in range(int(ms.ntrees)):
        t = yp[0][i]

        sid = int(t.sid)
        if sid <= 0:
            continue
        yp_tree_id = int(t.id)

        if yp_tree_id <= 0:
            identifier, tree_number = new_reference_tree_identity(stand)
            yp_tree_id = tree_number
            t.id = float(tree_number)
            idx = None
            storey = int(Storey.UNSET)
        else:
            idx = find_non_sapling_reference_tree_index(rt, sid, yp_tree_id)

            if idx is None:
                identifier, tree_number = new_reference_tree_identity(stand)
                yp_tree_id = tree_number
                t.id = float(tree_number)
                storey = int(Storey.UNSET)
            else:
                identifier = str(rt.identifier[idx])
                tree_number = int(rt.tree_number[idx])
                storey = int(rt.storey[idx]) if int(rt.storey[idx]) >= 0 else int(Storey.UNSET)

        row = {
            "identifier": identifier,
            "tree_number": int(yp_tree_id),
            "stratum": str(int(sid)),
            "species": int(t.spe),
            "stems_per_ha": float(t.f),
            "origin": int(t.snt) - 1,
            "height": float(t.h),
            "breast_height_diameter": float(t.d13),
            "biological_age": float(t.age),
            "breast_height_age": float(t.age13),
            "sapling": False,
            "tree_category": "",
            "management_category": 1,
            "storey": int(storey),
            "basal_area": (float(t.ba) / 10000.0) if getattr(t, "ba", None) is not None else 0.0,
            "volume": float(t.vol) if getattr(t, "vol", None) is not None else 0.0,
        }

        if idx is None:
            rt.create(row)
        else:
            rt.update(row, idx)


def _build_reference_tree_update(*,
                                 identifier: str,
                                 tree_number: int,
                                 osid: int,
                                 species: TreeSpecies,
                                 category_code: str,
                                 stems_per_ha: float,
                                 origin_raw: float,
                                 height: float,
                                 diameter: float,
                                 age: float,
                                 age13: float,
                                 basal_area: float,
                                 volume: float,
                                 storey: int,
                                 ) -> dict[str, Any]:
    return {
        "identifier": identifier,
        "tree_number": tree_number,
        "stratum": str(osid),
        "species": int(species),
        "stems_per_ha": stems_per_ha,
        "origin": safe_origin(origin_raw),
        "height": height,
        "breast_height_diameter": diameter,
        "biological_age": age,
        "breast_height_age": age13,
        "sapling": True,
        "tree_category": category_code,
        "management_category": 1,
        "storey": storey,
        "basal_area": basal_area,
        "volume": volume,
    }


def sync_ut_to_reference_trees(stand: ForestStand) -> None:
    ms = stand.motti_state
    if ms is None or ms.buffers is None:
        return

    ut = ms.buffers.saplings
    rt = stand.reference_trees
    next_osid = next_osite_id(stand)

    for layer in range(10):
        for spe_name, internal_species in UT_SPECIES_FIELDS:
            s = getattr(ut[0][layer], spe_name)

            try:
                if float(s.year) == -1.0:
                    continue
            except TypeError:
                continue

            for cat_code, _ in UT_CATEGORIES:
                stems = float(getattr(s, f"f_{cat_code}", 0.0) or 0.0)
                if stems <= 0.0:
                    continue

                osid_raw = getattr(s, f"osid_{cat_code}", 0.0)
                osid = int(osid_raw)

                if osid <= 0:
                    osid = next_osid
                    next_osid += 1
                    setattr(s, f"osid_{cat_code}", float(osid))

                idx = find_sapling_reference_tree_index(rt, osid)
                if idx is None:
                    identifier, tree_number = new_reference_tree_identity(stand)
                    storey = storey_from_layer(stand, layer)
                else:
                    identifier = str(rt.identifier[idx])
                    tree_number = int(rt.tree_number[idx])

                    existing_storey = int(rt.storey[idx])
                    if existing_storey >= 0:
                        storey = existing_storey
                    else:
                        storey = storey_from_layer(stand, layer)

                row = _build_reference_tree_update(identifier=identifier,
                                                   tree_number=tree_number,
                                                   osid=osid,
                                                   species=internal_species,
                                                   category_code="0",  # Small tree
                                                   stems_per_ha=stems,
                                                   origin_raw=float(getattr(s, f"N_{cat_code}", -1.0) or -1.0),
                                                   height=float(getattr(s, f"h_{cat_code}", 0.0) or 0.0),
                                                   diameter=float(getattr(s, f"d_{cat_code}", 0.0) or 0.0),
                                                   age=float(getattr(s, f"age_{cat_code}", 0.0) or 0.0),
                                                   age13=float(getattr(s, f"age13_{cat_code}", 0.0) or 0.0),
                                                   basal_area=float(getattr(s, f"g_{cat_code}", 0.0) or 0.0),
                                                   volume=float(getattr(s, f"v_{cat_code}", 0.0) or 0.0),
                                                   storey=storey,
                                                   )

                if idx is None:
                    rt.create(row)
                else:
                    rt.update(row, idx)


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


def _spedom(rt: ReferenceTrees) -> int:
    """
    Returns dominant species from Motti species.

    Prefer basal area totals; if BA totals are all zero/missing, fall back to stems/ha.
    If trees are empty fall back to PINE, we need to give valid value for growth.
    """
    if rt.size == 0:
        return TreeSpecies.PINE

    # Convert species to Motti codes (will raise if invalid)
    spe_codes = [convert_species(TreeSpecies(int(s))) for s in rt.species]

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
        if species < 0:
            continue

        biological_age = float(np.nan_to_num(strata.biological_age[i], nan=0.0))
        basal_area = float(np.nan_to_num(strata.basal_area[i], nan=0.0))
        stems_main = float(np.nan_to_num(strata.stems_per_ha[i], nan=0.0))
        mean_height = float(np.nan_to_num(strata.mean_height[i], nan=0.0))
        mean_diameter = float(np.nan_to_num(strata.mean_diameter[i], nan=0.0))
        origin = float(strata.origin[i])

        storey = storey_to_motti(
            stand,
            i,
            Storey(int(strata.storey[i])),
            is_stratum_index=True,
        )

        stratum_sid = int(strata.stratum_number[i])
        if stratum_sid <= 0:
            stratum_sid = i + 1

        spe = float(convert_species(species))
        out.append({
            "spe": spe,
            "age": biological_age,
            "ba": basal_area,
            "f": stems_main,
            "h": mean_height,
            "hw": mean_height,
            "d": mean_diameter,
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

    candidate_idx: list[int] = []
    for i in range(strata.size):
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
    rest_idx = [i for i in merge_idx if i != base_idx]

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


def _prune_promoted_sapling_reference_trees(stand: ForestStand) -> None:
    """
    Delete old sapling RFs if SID exists in YP vector.
    """
    ms = stand.motti_state
    rt = stand.reference_trees

    if ms is None or ms.yp is None or rt.size == 0:
        return

    yp_strata: set[int] = set()
    for i in range(int(ms.ntrees)):
        t = ms.yp[0][i]
        sid = int(t.sid)
        if sid > 0:
            yp_strata.add(sid)

    if not yp_strata:
        return

    delete_idx: list[int] = []
    for i in range(rt.size):
        if not bool(rt.sapling[i]):
            continue

        sid = int(rt.stratum[i])
        if sid > 0 and sid in yp_strata:
            delete_idx.append(i)

    if delete_idx:
        rt.delete(np.array(delete_idx, dtype=int))


def _prune_reference_trees_not_in_yp(stand: ForestStand) -> None:
    """
    Keep only ReferenceTrees that have a live in the YP vector.
    Used after Motti4Init init.
    """
    rt = stand.reference_trees
    ms = stand.motti_state

    if rt.size == 0:
        return

    live_yp: set[tuple[int, int]] = set()
    if ms is not None and ms.yp is not None:
        for i in range(int(ms.ntrees)):
            t = ms.yp[0][i]
            sid = int(t.sid)
            tree_id = int(t.id)
            if sid > 0 and tree_id > 0:
                live_yp.add((sid, tree_id))

    delete_idx: list[int] = []
    for i in range(rt.size):
        sid = int(rt.stratum[i])
        try:
            tree_number = int(rt.tree_number[i])
        except (TypeError, ValueError):
            tree_number = -1

        if sid <= 0 or tree_number <= 0 or (sid, tree_number) not in live_yp:
            delete_idx.append(i)

    if delete_idx:
        rt.delete(np.array(delete_idx, dtype=int))


def reconcile_reference_trees_from_motti(stand: ForestStand, *, init_mode: bool = False) -> None:
    sync_yp_to_reference_trees(stand)
    _prune_promoted_sapling_reference_trees(stand)

    if init_mode:
        _prune_reference_trees_not_in_yp(stand)

    sync_ut_to_reference_trees(stand)
    prune_reference_trees_not_in_motti(stand)


def ensure_state(stand: ForestStand,
                 step: int,
                 sim_year: int,
                 use_dll_site_convert: bool = True):
    """Initialize and attach persistent MottiState to stand if missing."""
    if stand.motti_state is not None:
        return stand.motti_state

    rt = stand.reference_trees

    n = rt.size

    spedom = _spedom(stand.reference_trees)

    y_km, x_km = auto_euref_km(stand.geo_location[0] if stand.geo_location is not None else None,
                               stand.geo_location[1] if stand.geo_location is not None else None)

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
        mty=stand.site_type_category.value if stand.site_type_category is not None else 0,
        verl=stand.tax_class if stand.tax_class is not None else 0,
        verlt=stand.tax_class_reduction if stand.tax_class_reduction is not None else 0,
        xt_regen=((stand.start_time - stand.artificial_regeneration_year)
                  if stand.artificial_regeneration_year is not None else stand.start_time),
        xt_muok=((stand.start_time - stand.soil_surface_preparation_year)
                 if stand.soil_surface_preparation_year is not None else stand.start_time),
        xt_raiv=((stand.start_time - stand.regeneration_area_cleaning_year)
                 if stand.regeneration_area_cleaning_year is not None else stand.start_time),
        sid=stand.stand_id or 0,
        fthin=bool(stand.method_of_last_cutting),
        xt_thin=stand.method_of_last_cutting or stand.cutting_year or 0,
        xt_fert=((stand.start_time - stand.fertilization_year)
                 if stand.fertilization_year is not None else stand.start_time),
        xt_thoit=((stand.start_time - stand.young_stand_tending_year)
                  if stand.young_stand_tending_year is not None else stand.start_time),
        drain=stand.drainage_category.value if stand.drainage_category is not None else 0,
        xt_ndrain=((stand.start_time - stand.drainage_year)
                   if stand.drainage_year is not None else stand.start_time),
        alr=stand.soil_peatland_category.value if stand.soil_peatland_category is not None else 0,
        year=sim_year - stand.start_year,
        step=step,
        convert_mela_site=use_dll_site_convert,
        spedom=spedom,
        spedom2=spedom,
        nstorey=1.0,
        gstorey=1.0,
    )

    # TODO: Is this right?
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
    spe_vec = [convert_species(TreeSpecies(int(s))) for s in rt.species]

    stratum_ids = [
        int(v) if v > 0 else (stand.stand_id or (idx + 1))
        for idx, v in enumerate(rt.stratum)
    ]
    storey_vec = [storey_to_motti(stand, idx, Storey(int(rt.storey[idx]))) for idx in range(n)]
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
            stratum_ids,
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

    if MottiState is not None:
        stand.motti_state = MottiState(
            yy=yy,
            yp=yp,
            ntrees=ntrees,
            buffers=buffers,
        )
    else:
        ms = MottiState()
        ms.yy = yy
        ms.yp = yp
        ms.ntrees = ntrees
        ms.buffers = buffers
        stand.motti_state = ms

    reconcile_reference_trees_from_motti(stand, init_mode=True)

    return stand.motti_state


def _mark_motti_yp_as_seed_trees(stand: ForestStand, *, tree_class: int = 3) -> bool:
    """Mark all currently live YP trees as seed trees for Motti.

    Motti's YP vector field at documented index 38 is exposed in the
    CFFI wrapper as ``storie``. For seed-tree cutting the remaining trees
    must have puuluokka/tree class 3 before Motti4AfterSeedtreeCutting is
    called.
    """

    ms = stand.motti_state
    if ms is None or ms.yp is None:
        return False

    changed = False
    for i in range(int(ms.ntrees)):
        t = ms.yp[0][i]
        if float(t.f) <= 0.0:
            continue
        if float(t.storie) != float(tree_class):
            t.storie = float(tree_class)
            changed = True

    return changed


def after_seedtree_cutting_in_motti(stand: ForestStand, *, tree_class: int = 3) -> None:
    """Run Motti seed-tree cutting post-processing and sync Python vectors."""

    ms = stand.motti_state
    if ms is None or ms.yp is None or ms.buffers is None:
        return

    _mark_motti_yp_as_seed_trees(stand, tree_class=tree_class)

    ms.ntrees = Motti4DLL.after_seedtree_cutting_with_state(
        ms.yy,
        ms.yp,
        int(ms.ntrees),
        ms.buffers,
    )

    reconcile_reference_trees_from_motti(stand)


def _refresh_reference_trees_from_motti_after_yp_change(stand: ForestStand) -> None:
    """
    Rebuild Motti internal state after yp edits, run grow(step=0) and
    then synchronize ReferenceTrees from yp/ut.
    """
    ms = stand.motti_state
    if ms is None or ms.yp is None or ms.buffers is None:
        return

    Motti4DLL.grow_with_state(ms, step=0)

    reconcile_reference_trees_from_motti(stand)


def _reduce_motti_yp_by_removed_reference_trees(stand: ForestStand, removed_f: npt.NDArray[np.float64]) -> bool:
    ms = stand.motti_state
    trees = stand.reference_trees
    if ms is None or ms.yp is None or trees.size == 0:
        return False

    changed = False
    for idx, delta in enumerate(removed_f):
        if delta <= 0.0:
            continue
        if bool(trees.sapling[idx]):
            continue

        sid = int(trees.stratum[idx])
        if sid <= 0:
            continue

        tree_number = int(trees.tree_number[idx])
        if tree_number <= 0:
            continue

        for i in range(int(ms.ntrees)):
            t = ms.yp[0][i]
            if int(t.sid) != sid:
                continue
            if int(t.id) != tree_number:
                continue

            new_f = max(float(t.f) - float(delta), 0.0)
            if new_f != float(t.f):
                t.f = float(new_f)
                changed = True
            break

    return changed


def apply_motti_yp_reduction_from_removed_reference_trees(stand: ForestStand,
                                                          removed_f: np.ndarray,
                                                          *,
                                                          refresh: bool = True,
                                                          ) -> bool:
    """
    Generic helper for treatments that reduce tree amounts.
    This helper maps removed trees to the yp vector via the shared stratum/sid,
    optionally runs a zero-step Motti refresh, and synchronizes ReferenceTrees
    back from the refreshed Motti state.
    """
    changed = _reduce_motti_yp_by_removed_reference_trees(stand, removed_f)
    if changed and refresh:
        _refresh_reference_trees_from_motti_after_yp_change(stand)
    return changed


def _collect_live_motti_keys(stand: ForestStand) -> set[tuple[str, int, int | None]]:
    live: set[tuple[str, int, int | None]] = set()

    ms = stand.motti_state
    if ms is None:
        return live

    if ms.yp is not None:
        for i in range(int(ms.ntrees)):
            t = ms.yp[0][i]

            sid = int(t.sid)
            tree_id = int(t.id)

            if sid > 0 and tree_id > 0:
                live.add(("yp", sid, tree_id))

    if ms.buffers is not None:
        ut = ms.buffers.saplings
        for layer in range(10):
            for spe_name, _ in UT_SPECIES_FIELDS:
                s: Motti4SaplingStratum = getattr(ut[0][layer], spe_name)
                for cat_code, _ in UT_CATEGORIES:
                    stems = float(getattr(s, f"f_{cat_code}"))
                    if stems <= 0.0:
                        continue
                    osid = int(getattr(s, f"osid_{cat_code}"))
                    if osid > 0:
                        live.add(("ut", osid, None))

    return live


def prune_reference_trees_not_in_motti(stand: ForestStand) -> None:

    rt = stand.reference_trees
    if rt.size == 0:
        return

    live = _collect_live_motti_keys(stand)
    delete_idx = []

    for i, value in enumerate(rt.stratum):
        sid = int(value)
        if sid <= 0:
            continue

        key: tuple[str, int, int | None]
        if bool(rt.sapling[i]):
            key = ("ut", sid, None)
        else:
            try:
                tree_number = int(rt.tree_number[i])
            except (TypeError, ValueError):
                tree_number = -1
            key = ("yp", sid, tree_number)

        if key not in live:
            delete_idx.append(i)

    if delete_idx:
        rt.delete(np.array(delete_idx, dtype=int))
