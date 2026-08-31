from typing import Any, Optional

import numpy as np
from lukefi.metsi.data.conversion import internal2motti
from lukefi.metsi.data.enums.internal import CRS, CuttingMethod, Storey, TreeSpecies
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.motti.motti_types import MottiState
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.domain.natural_processes import motti_util
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import Motti4DLL


# NOTE: Why not use what is in enum-modules?
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

    # NOTE: We lose all original data info about species, id_number and generated tree count, but does it matter?
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
    Prefer basal area totals; if BA totals are all zero/missing, fall back to stems/ha.
    If trees are empty fall back to PINE, we need to give valid value for growth.

    NOTE: This should be generalized to ForestStand.spedom like solution.
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
    """
    if strata is None:
        strata = stand.tree_strata

    if strata.size == 0:
        return []

    out: list[dict[str, float]] = []

    for i in range(min(strata.size, 10)):
        species = TreeSpecies(strata.species[i].item())
        if species <= TreeSpecies.TREELESS:
            continue # Skips strata if no trees.

        _basal_area = float(np.nan_to_num(strata.basal_area[i], nan=0.0))
        basal_area = _basal_area if _basal_area > 0.001 else 0.0
        biological_age = float(np.nan_to_num(strata.biological_age[i], nan=0.0))
        stems_main = float(np.nan_to_num(strata.stems_per_ha[i], nan=0.0))
        mean_height = float(np.nan_to_num(strata.mean_height[i], nan=0.0))
        mean_diameter = float(np.nan_to_num(strata.mean_diameter[i], nan=0.0))
        origin = float(strata.origin[i].item())
        stratum_sid = float(strata.stratum_number[i].item())
        spe = float(internal2motti.convert_species(species))
        storey = _storey_to_motti(
            stand,
            i,
            Storey(strata.storey[i].item()),
            is_stratum_index=True)

        out.append({
            "spe": spe,
            "age": biological_age,
            "ba": basal_area,
            "f": stems_main,
            "h": 0.0,
            "hw": mean_height,
            "d": 0.0,
            "dg": mean_diameter,
            "storey": storey,
            "st": origin,
            "sid": stratum_sid,
        })

    return out


def _compress_strata_for_motti(stand: ForestStand, max_strata: int = 10) -> TreeStrata:
    """
    If there are more than max_strata strata, merge retention stata into one so the count becomes max_strata.

    Candidate retention strata for merge is:
      - number_of_generated_trees == 1
      - storey == SPARE

    Values for compression strata are calculated from reference trees
        that share the same stratum number as the compressed strata

    Merged result:
      - species = species whose candidate strata have the highest total stems_per_ha
      - mean_height = avg
      - mean_diameter = avg
      - stems_per_ha = sum
      - storey / origin / stratum_rank / stratum_number / identifier = from base row

    If there are not enough merge candidates, return original strata unchanged.
    """
    strata: TreeStrata = stand.tree_strata

    if strata.size <= max_strata:
        return strata

    excess = strata.size - max_strata
    if excess <= 0:
        return strata

    retention_idx: list[int] = []
    for i in range(strata.size):
        gen_n = strata.number_of_generated_trees[i].item()
        storey = strata.storey[i].item()
        if gen_n == 1 and storey == Storey.SPARE:
            retention_idx.append(i)

    needed = excess + 1
    if len(retention_idx) < needed:
        return strata  # fallback: current truncation behavior stays

    # take exactly as many as needed; simplest and least invasive
    merge_idx = retention_idx[:needed]

    # species totals by stems_per_ha -> choose dominant/base species
    stems_by_species: dict[int, float] = {}
    for i in merge_idx:
        species = strata.species[i].item()
        stems = strata.stems_per_ha[i].item()
        stems_by_species[species] = stems_by_species.get(species, 0.0) + stems

    base_species = max(stems_by_species.items(), key=lambda kv: kv[1])[0]

    # (choose from needed retention indices) the base row as first row of the major species
    base_idx = next(i for i in merge_idx if strata.species[i].item() == base_species)
    rest_idx = [i for i in merge_idx if i != base_idx]

    out = strata[:]

    # merged numeric values
    out.stems_per_ha[base_idx] = float(np.nansum(out.stems_per_ha[merge_idx]))

    # calculate the base compression strata values from same stratum number trees as merge_idx
    merge_mask = np.isin(stand.reference_trees.stratum, strata.stratum_number[merge_idx])
    merge_trees = stand.reference_trees[merge_mask]
    divider = np.sum(merge_trees.basal_area * merge_trees.stems_per_ha)
    out.mean_diameter[base_idx] = (
                (np.sum(
                    merge_trees.stems_per_ha *
                    merge_trees.basal_area *
                    merge_trees.breast_height_diameter)) / divider) if divider > 0.0 else None
    out.mean_height[base_idx] = (
                (np.sum(
                    merge_trees.stems_per_ha *
                    merge_trees.basal_area *
                    merge_trees.height)) / divider) if divider > 0.0 else None
    out.biological_age[base_idx] = (
                (np.sum(
                    merge_trees.stems_per_ha *
                    merge_trees.basal_area *
                    merge_trees.biological_age)) / divider) if divider > 0.0 else None
    out.breast_height_age[base_idx] = (
                (np.sum(
                    merge_trees.stems_per_ha *
                    merge_trees.basal_area *
                    merge_trees.breast_height_age)) / divider) if divider > 0.0 else None
    out.sapling_stems_per_ha[base_idx] = float(np.nansum(out.sapling_stems_per_ha[merge_idx]))

    # force species to same
    out.species[base_idx] = base_species

    # remove compressed strata
    if rest_idx:
        out.delete(rest_idx)

    # update the stratum values of reference trees after compression
    stand.reference_trees.stratum[merge_mask] = strata.stratum_number[base_idx]

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

    compressed_strata = _compress_strata_for_motti(stand, max_strata=10)

    rt = stand.reference_trees
    n = len(rt)

    ids = rt.tree_number.astype(int).copy()
    stems = np.nan_to_num(rt.stems_per_ha, nan=0.0)
    d13 = np.nan_to_num(rt.breast_height_diameter, nan=0.0)
    h = np.nan_to_num(rt.height, nan=0.0)
    age = np.nan_to_num(rt.biological_age, nan=0.0)
    age13 = np.nan_to_num(rt.breast_height_age, nan=0.0)

    origin = rt.origin
    spe_vec = [internal2motti.convert_species(TreeSpecies(int(s))) for s in rt.species]

    stratum_ids = rt.stratum.tolist()
    if -1 in stratum_ids:
        raise ValueError("ReferenceTrees contains stratum_number=-1, which is invalid for Motti initialization.")
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
            "snt": int(o + 1),
            "storie": float(storey),

        }
        for i, sid, f, d, hh, sp, a, a13, o, storey in zip(
            ids,
            stratum_ids,  # original osite_id
            stems,
            d13,
            h,
            spe_vec,
            age,
            age13,
            origin,
            storey_vec,
        )
    ]
    yp, ntrees = Motti4DLL.new_trees(trees_py)

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
