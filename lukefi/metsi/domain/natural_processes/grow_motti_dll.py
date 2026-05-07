import os
from typing import Any, Optional, Dict, Union, Iterable
from pathlib import Path
import numpy as np
import numpy.typing as npt
from lukefi.metsi.data.conversion.internal2motti import convert_species
from lukefi.metsi.domain.natural_processes.motti_dll_wrapper import (
    Motti4DLL,
    GrowthDeltas,
)
from lukefi.metsi.data.enums.internal import (
    LandUseCategory,
    TreeSpecies,
    Storey,
)
from lukefi.metsi.data.model import ForestStand, MottiState
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.domain.natural_processes.util import (
    update_stand_growth, safe_storey_value,
    UT_SPECIES_FIELDS,
    UT_CATEGORIES,
    parse_int_id,
    next_osite_id,
    safe_origin,
    new_reference_tree_identity,
    storey_from_layer,
    find_non_sapling_reference_tree_index,
    find_sapling_reference_tree_index,
    storey_to_motti
)
from lukefi.metsi.domain.natural_processes.natural_process_wrapper import natural_process_transition
from lukefi.metsi.sim.collected_data import OpTuple


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
    if strata is None or strata.size <= max_strata:
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
    out.species[base_idx] = int(base_species)

    if rest_idx:
        out.delete(np.array(rest_idx, dtype=int))

    return out


def _build_reference_tree_update(
    *,
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
        "tree_number": int(tree_number),
        "stratum": str(int(osid)),
        "species": int(species),
        "stems_per_ha": float(stems_per_ha),
        "origin": safe_origin(origin_raw),
        "height": float(height),
        "breast_height_diameter": float(diameter),
        "biological_age": float(age),
        "breast_height_age": float(age13),
        "sapling": True,
        "tree_category": category_code,
        "management_category": 1,
        "storey": int(storey),
        "basal_area": float(basal_area),
        "volume": float(volume),
    }


def _build_motti_strata_py(
    stand: ForestStand,
    strata: TreeStrata | None = None,
) -> list[dict]:
    """
    Convert given TreeStrata into Python dicts for Motti4Strata.
    If strata is not given, use stand.tree_strata.

    Uncertain fields:
      hw -> temporary fallback to mean_height
      dg -> temporary fallback to mean_diameter
      st -> temporary dummy 0.0
    """
    if strata is None:
        strata = getattr(stand, "tree_strata", None)

    if strata is None or strata.size == 0:
        return []

    out: list[dict] = []

    for i in range(min(strata.size, 10)):
        species = TreeSpecies(int(strata.species[i]))
        if species < 0:
            continue

        biological_age = float(np.nan_to_num(strata.biological_age[i], nan=0.0))
        basal_area = float(np.nan_to_num(strata.basal_area[i], nan=0.0))
        stems_main = float(np.nan_to_num(strata.stems_per_ha[i], nan=0.0))
        mean_height = float(np.nan_to_num(strata.mean_height[i], nan=0.0))
        mean_diameter = float(np.nan_to_num(strata.mean_diameter[i], nan=0.0))
        origin = safe_storey_value(strata.origin[i])

        storey = storey_to_motti(
            stand,
            i,
            strata.storey[i],
            is_stratum_index=True,
        )

        stratum_sid = parse_int_id(strata.stratum_number[i])
        if stratum_sid is None:
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


def _spedom(rt: ReferenceTrees | None) -> int:
    """
    Returns dominant species from Motti species.

    Prefer basal area totals; if BA totals are all zero/missing, fall back to stems/ha.
    If trees are empty fall back to PINE, we need to give valid value for growth.
    """
    if rt is None:
        return TreeSpecies.PINE
    n = rt.size
    if n == 0:
        return TreeSpecies.PINE

    # Convert species to Motti codes (will raise if invalid)
    spe_codes =[convert_species(TreeSpecies(int(s))) for s in rt.species]

    # Basal area per tree: stems_per_ha * π * (0.5 * d_cm * 0.01 m/cm)^2
    d_cm = np.nan_to_num(rt.breast_height_diameter, nan=0.0)
    f_ha = np.nan_to_num(rt.stems_per_ha, nan=0.0)
    ba_per_tree = f_ha * np.pi * (0.5 * d_cm * 0.01) ** 2  # m²/ha contribution

    # Sum BA per species code
    per: Dict[int, float] = {}
    for code, ba in zip(spe_codes, ba_per_tree.tolist()):
        per[code] = per.get(code, 0.0) + float(ba)

    use_basal = any(v > 0.0 for v in per.values())
    if not use_basal:
        per.clear()
        # Fallback: stems/ha totals per species
        for code, stems in zip(spe_codes, f_ha.tolist()):
            per[code] = per.get(code, 0.0) + float(stems)

    if not per:
        return TreeSpecies.PINE

    return max(per.items(), key=lambda kv: kv[1])[0]


def _strip_tree_strata(stand: ForestStand):
    """
    Clear tree information from  strata
    """
    if stand.tree_strata is None or stand.tree_strata.size == 0:
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


def _reduce_motti_yp_by_removed_reference_trees(stand: ForestStand, removed_f: np.ndarray) -> bool:
    ms = getattr(stand, "motti_state", None)
    rt = getattr(stand, "reference_trees", None)
    if ms is None or ms.yp is None or rt is None or rt.size == 0:
        return False

    changed = False
    for idx, delta in enumerate(np.asarray(removed_f, dtype=float).tolist()):
        if delta <= 0.0:
            continue
        if bool(rt.sapling[idx]):
            continue

        sid = parse_int_id(rt.stratum[idx])
        if sid is None:
            continue

        tree_number = int(rt.tree_number[idx])
        if tree_number <= 0:
            continue

        for i in range(int(ms.ntrees)):
            t = ms.yp[0][i]
            if parse_int_id(getattr(t, "sid", 0)) != sid:
                continue
            if parse_int_id(getattr(t, "id", 0)) != tree_number:
                continue

            new_f = max(float(t.f) - float(delta), 0.0)
            if new_f != float(t.f):
                t.f = float(new_f)
                changed = True
            break

    return changed


def sync_ut_to_reference_trees(stand: ForestStand) -> None:
    ms = getattr(stand, "motti_state", None)
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

            for cat_code, _cat_label in UT_CATEGORIES:
                stems = float(getattr(s, f"f_{cat_code}", 0.0) or 0.0)
                if stems <= 0.0:
                    continue

                osid_raw = getattr(s, f"osid_{cat_code}", 0.0)
                osid = parse_int_id(osid_raw)

                if osid is None:
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

                row = _build_reference_tree_update(
                    identifier=identifier,
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


def sync_yp_to_reference_trees(stand: ForestStand) -> None:
    ms = getattr(stand, "motti_state", None)
    if ms is None or ms.yp is None:
        return

    yp = ms.yp
    rt = stand.reference_trees

    for i in range(int(ms.ntrees)):
        t = yp[0][i]

        sid = parse_int_id(getattr(t, "sid", 0))
        if sid is None:
            continue
        yp_tree_id = parse_int_id(getattr(t, "id", 0))

        if yp_tree_id is None:
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
            "origin": safe_origin(int(t.snt) - 1),
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


def prune_reference_trees_not_in_yp(stand: ForestStand) -> None:
    """
    Keep only ReferenceTrees that have a live in the YP vector.
    Used after Motti4Init init.
    """
    rt = stand.reference_trees
    ms = getattr(stand, "motti_state", None)

    if rt is None or rt.size == 0:
        return

    live_yp: set[tuple[int, int]] = set()
    if ms is not None and ms.yp is not None:
        for i in range(int(ms.ntrees)):
            t = ms.yp[0][i]
            sid = parse_int_id(getattr(t, "sid", 0))
            tree_id = parse_int_id(getattr(t, "id", 0))
            if sid is not None and tree_id is not None:
                live_yp.add((sid, tree_id))

    delete_idx: list[int] = []
    for i in range(rt.size):
        sid = parse_int_id(rt.stratum[i])
        try:
            tree_number = int(rt.tree_number[i])
        except (TypeError, ValueError):
            tree_number = -1

        if sid is None or tree_number <= 0 or (sid, tree_number) not in live_yp:
            delete_idx.append(i)

    if delete_idx:
        rt.delete(np.array(delete_idx, dtype=int))


def reconcile_reference_trees_from_motti(stand: ForestStand, *, init_mode: bool = False) -> None:
    sync_yp_to_reference_trees(stand)
    prune_promoted_sapling_reference_trees(stand)

    if init_mode:
        prune_reference_trees_not_in_yp(stand)

    sync_ut_to_reference_trees(stand)
    prune_reference_trees_not_in_motti(stand)


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


def find_repo_root(start: Path) -> Optional[Path]:
    """
    Walk up from 'start' to find a repository root by markers:
    - a directory that contains 'data/motti'
    - or has a '.git' directory
    - or has a 'pyproject.toml' file
    """
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "data" / "motti").exists():
            return p
        if (p / ".git").exists():
            return p
        if (p / "pyproject.toml").exists():
            return p
    return None


def default_data_dir() -> Path:
    """
    Resolve default data_dir as {repository_root}/data/motti,
    with optional override via MOTTI_DATA_DIR.
    """
    env = os.environ.get("MOTTI_DATA_DIR")
    if env:
        return Path(os.path.expanduser(os.path.expandvars(env))).resolve()
    repo = find_repo_root(Path.cwd())
    base = repo if repo else Path.cwd()
    return (base / "data" / "motti").resolve()


def resolve_dir_or_file(path_like: Optional[str | Path]) -> Path:
    """
    Turn a user-provided path into an absolute Path. If None, use default.
    """
    if path_like is None:
        return default_data_dir()
    p = Path(os.path.expanduser(os.path.expandvars(str(path_like))))
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


# -------- vectorized predictor --------
class MottiDLLPredictor:
    """
    SoA-based predictor feeding the Motti DLL. Builds C tree buffers from vector arrays.
    """

    def __init__(
        self,
        stand: ForestStand,
        data_dir: Optional[str] = None,
        use_dll_site_convert: bool = True,
        dll: Optional["Motti4DLL"] = None,
    ) -> None:
        self.stand = stand
        self.use_dll_site_convert = use_dll_site_convert

        if dll is not None:
            self.dll = dll
        else:

            # Resolve given path or default to {repo_root}/data/motti
            data_dir_path = resolve_dir_or_file(data_dir)

            so_path = resolve_shared_object(data_dir_path)
            self.dll = Motti4DLL(so_path, data_dir=str(data_dir_path))

    # ---- stand/site properties ----
    @property
    def year(self) -> float:
        y = getattr(self.stand, "start_year", None)
        return float(y) if y is not None else 2010.0

    @property
    def get_y(self) -> float | None:
        if self.stand and self.stand.geo_location:
            return self.stand.geo_location[0]
        return None

    @property
    def get_x(self) -> float | None:
        if self.stand and self.stand.geo_location:
            return self.stand.geo_location[1]
        return None

    @property
    def get_z(self) -> float:
        if self.stand and self.stand.geo_location:
            z = self.stand.geo_location[2]
            if z is None or z == 0.0:
                return -1.0
            return float(z)
        return -1.0

    @property
    def lake(self) -> float:
        v = getattr(self.stand, "lake_effect", 0.0)
        return float(v if v is not None else 0.0)

    @property
    def sea(self) -> float:
        v = getattr(self.stand, "sea_effect", 0.0)
        return float(v if v is not None else 0.0)

    @property
    def mal(self) -> int:
        luc = getattr(self.stand, "land_use_category", None)
        return int(luc.value) if luc is not None else 0

    @property
    def mty(self) -> int:
        st = getattr(self.stand, "site_type_category", None)
        return int(st.value) if st is not None else 0

    @property
    def alr(self) -> int:
        s = getattr(self.stand, "soil_peatland_category", None)
        return int(s.value) if s is not None else 0

    @property
    def verl(self) -> int:
        v = getattr(self.stand, "tax_class", None)
        return int(v) if v is not None else 0

    @property
    def verlt(self) -> int:
        v = getattr(self.stand, "tax_class_reduction", None)
        return int(v) if v is not None else 0

    @property
    def xt_regen(self) -> int:
        art = self.stand.artificial_regeneration_year
        return (self.stand.start_time - art) if art \
            is not None else self.stand.start_time

    @property
    def xt_muok(self) -> int:
        soil = self.stand.soil_surface_preparation_year
        return (self.stand.start_time - soil) if soil \
            is not None else self.stand.start_time

    @property
    def xt_raiv(self) -> int:
        reg = self.stand.regeneration_area_cleaning_year
        return (self.stand.start_time - reg) if reg \
            is not None else self.stand.start_time

    @property
    def sid(self) -> int:
        return self.stand.stand_id or 0

    @property
    def fthin(self) -> bool:
        return bool(self.stand.method_of_last_cutting)

    @property
    def xt_thin(self) -> int:
        return (self.stand.method_of_last_cutting or self.stand.cutting_year or 0)

    @property
    def xt_fert(self) -> int:
        return (self.stand.start_time - self.stand.fertilization_year) if self.stand.fertilization_year \
            is not None else self.stand.start_time

    @property
    def xt_thoit(self) -> int:
        return (self.stand.start_time - self.stand.young_stand_tending_year) if self.stand.young_stand_tending_year \
            is not None else self.stand.start_time

    @property
    def drain(self) -> int:

        if not self.stand.drainage_category:
            return 0
        return self.stand.drainage_category.value

    @property
    def xt_ndrain(self) -> int:
        return (self.stand.start_time - self.stand.drainage_year) if self.stand.drainage_year \
            is not None else self.stand.start_time

    def ensure_state(self, step: int, sim_year: int):
        """Initialize and attach persistent MottiState to stand if missing."""
        if getattr(self.stand, "motti_state", None) is not None:
            return self.stand.motti_state

        rt = self.stand.reference_trees

        n = rt.size

        spedom = _spedom(self.stand.reference_trees)

        y_km, x_km = auto_euref_km(self.get_y, self.get_x)
        yy = self.dll.new_site(
            Y=y_km,
            X=x_km,
            Z=self.get_z,
            lake=self.lake,
            sea=self.sea,
            mal=self.mal,
            mty=self.mty,
            verl=self.verl,
            verlt=self.verlt,
            xt_regen=self.xt_regen,
            xt_muok=self.xt_muok,
            xt_raiv=self.xt_raiv,
            sid=self.sid,
            fthin=self.fthin,
            xt_thin=self.xt_thin,
            xt_fert=self.xt_fert,
            xt_thoit=self.xt_thoit,
            drain=self.drain,
            xt_ndrain=self.xt_ndrain,
            alr=self.alr,
            year=sim_year - self.stand.start_year,
            step=step,
            convert_mela_site=self.use_dll_site_convert,
            spedom=spedom,
            spedom2=spedom,
            nstorey=1.0,
            gstorey=1.0,
        )

        rt.tree_number = np.arange(1, n + 1, dtype=rt.tree_number.dtype)
        ids = rt.tree_number.astype(int).copy()

        stems = np.nan_to_num(rt.stems_per_ha, nan=0.0)
        d13 = np.nan_to_num(rt.breast_height_diameter, nan=0.0)
        h = np.nan_to_num(rt.height, nan=0.0)
        age = np.nan_to_num(rt.biological_age, nan=0.0)
        age13 = np.nan_to_num(rt.breast_height_age, nan=0.0)
        cr = np.nan_to_num(getattr(rt, "crown_ratio", np.zeros(n, dtype=float)), nan=0.0)
        origin = np.nan_to_num(getattr(rt, "origin", np.zeros(n, dtype=float)), nan=0.0)
        spe_vec = [convert_species(TreeSpecies(int(s))) for s in rt.species]

        stratum_ids = [
            parse_int_id(v) or (self.stand.stand_id or (idx + 1))
            for idx, v in enumerate(rt.stratum.tolist())
        ]
        storey_vec = np.asarray(
            [
                storey_to_motti(self.stand, idx, rt.storey[idx])
                for idx in range(n)
            ],
            dtype=int,
        )
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
                ids.tolist(),
                stratum_ids,
                stems.tolist(),
                d13.tolist(),
                h.tolist(),
                spe_vec,
                age.tolist(),
                age13.tolist(),
                cr.tolist(),
                origin.astype(int).tolist(),
                storey_vec.tolist(),
            )
        ]
        yp, ntrees = self.dll.new_trees(trees_py)

        compressed_strata = _compress_strata_for_motti(self.stand.tree_strata, max_strata=10)
        strata_py = _build_motti_strata_py(self.stand, compressed_strata)

        yo = self.dll.new_strata(strata_py)

        buffers = self.dll.alloc_state_buffers(ctrl=None)

        ntrees = self.dll.initialize_with_state(
            yo=yo,
            yy=yy,
            yp=yp,
            numtrees=int(ntrees),
            buffers=buffers,
        )

        _strip_tree_strata(self.stand)

        if MottiState is not None:
            self.stand.motti_state = MottiState(
                dll=self.dll,
                yy=yy,
                yp=yp,
                ntrees=int(ntrees),
                buffers=buffers,
                signature=tuple(ids.tolist()),
            )
        else:
            ms = MottiState()
            ms.dll = self.dll
            ms.yy = yy
            ms.yp = yp
            ms.ntrees = int(ntrees)
            ms.buffers = buffers
            ms.signature = tuple(ids.tolist())
            self.stand.motti_state = ms

        reconcile_reference_trees_from_motti(self.stand, init_mode=True)

        return self.stand.motti_state

    def evolve(self, step: int = 5, sim_year: int = 0) -> GrowthDeltas:
        state = self.ensure_state(step=step, sim_year=sim_year)
        if state is None:
            return GrowthDeltas(tree_ids=[], tree_sids=[], trees_id=[], trees_ih=[], trees_if=[],
                                trees_age=[], trees_age13=[]
                                )

        state.yy.year = sim_year
        state.yy.step = step

        growth = self.dll.grow_with_state(
            state.yy,
            state.yp,
            int(state.ntrees),
            state.buffers,
            step=step,
        )

        state.ntrees = len(growth.tree_ids)

        return growth


# -------- DLL path resolver (same behavior as AoS helper) --------

def resolve_shared_object(p: Union[str, Path]) -> Path:
    """
    Resolve a Motti shared library inside a directory, or pass through an exact file path.
    Raises ValueError if p is None. Returns a Path (may be a directory if nothing matched).
    """
    if p is None:
        raise ValueError("data_dir must be provided (directory containing the Motti library).")

    p = Path(p)

    if p.is_file():
        return p

    candidates: Iterable[str] = (
        # Windows
        "mottisc.dll", "mottiue.dll",
        # Linux
        "libmottisc.so", "libmottiue.so", "mottisc.so", "mottiue.so",
    )
    for name in candidates:
        cand = p / name
        if cand.exists():
            return cand

    # No match found; return directory so downstream can raise a clear error when loading.
    return p


# -------- public API --------

@natural_process_transition
def grow_motti_dll_fn(input_: ForestStand, step: int = 5, /, **operation_parameters) -> OpTuple[ForestStand]:
    """
    Vector-only Motti grow:
      - Requires stand.reference_trees
      - Builds DLL input from SoA, runs growth, applies deltas vectorized
      - Prunes trees with stems_per_ha < 1.0 after update
    operation_parameters:
      - data_dir: path to folder/file for the Motti DLL (required unless a predictor is injected)
      - predictor: optional injected Motti4DLL wrapper (testing)
    """

    data_dir = operation_parameters.get("data_dir", None)
    predictor = operation_parameters.get("predictor", None)

    stand = input_

    sim_year: int = (stand.year - stand.start_year) or 0

    rt = stand.reference_trees

    if stand.land_use_category and stand.land_use_category >= LandUseCategory.WASTE_LAND:
        base_d = np.nan_to_num(rt.breast_height_diameter, nan=0.0)
        base_h = np.nan_to_num(rt.height, nan=0.0)
        base_f = np.nan_to_num(rt.stems_per_ha, nan=0.0)
        update_stand_growth(stand, base_d, base_h, base_f, step, False)
        return stand, []

    if predictor is None:
        resolved_dir = resolve_dir_or_file(data_dir)
        pred = MottiDLLPredictor(stand, data_dir=str(resolved_dir))
    else:
        pred = predictor

    pred.evolve(step=step, sim_year=sim_year)
    stand.year = (stand.year or 0) + step

    reconcile_reference_trees_from_motti(stand, init_mode=False)

    return stand, []


def refresh_reference_trees_from_motti_after_yp_change(stand: ForestStand) -> None:
    """
    Rebuild Motti internal state after yp edits, run grow(step=0) and
    then synchronize ReferenceTrees from yp/ut.
    """
    ms = stand.motti_state
    if ms is None or ms.yp is None or ms.buffers is None:
        return

    growth = ms.dll.grow_with_state(
        ms.yy,
        ms.yp,
        int(ms.ntrees),
        ms.buffers,
        step=0,
    )
    ms.ntrees = len(growth.tree_ids)

    reconcile_reference_trees_from_motti(stand)


def apply_motti_yp_reduction_from_removed_reference_trees(
    stand: ForestStand,
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
        refresh_reference_trees_from_motti_after_yp_change(stand)
    return changed


def mark_motti_yp_as_seed_trees(stand: ForestStand, *, tree_class: int = 3) -> bool:
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
        if float(getattr(t, "f", 0.0) or 0.0) <= 0.0:
            continue
        if float(getattr(t, "storie", 0.0) or 0.0) != float(tree_class):
            t.storie = float(tree_class)
            changed = True

    return changed


def after_seedtree_cutting_in_motti(stand: ForestStand, *, tree_class: int = 3) -> None:
    """Run Motti seed-tree cutting post-processing and sync Python vectors."""

    ms = stand.motti_state
    if ms is None or ms.yp is None or ms.buffers is None:
        return

    mark_motti_yp_as_seed_trees(stand, tree_class=tree_class)

    ms.ntrees = ms.dll.after_seedtree_cutting_with_state(
        ms.yy,
        ms.yp,
        int(ms.ntrees),
        ms.buffers,
    )

    reconcile_reference_trees_from_motti(stand)


def collect_live_motti_keys(stand: ForestStand) -> set[tuple[str, int, int | None]]:
    live: set[tuple[str, int, int | None]] = set()

    ms = stand.motti_state
    if ms is None:
        return live

    if ms.yp is not None:
        for i in range(int(ms.ntrees)):
            t = ms.yp[0][i]

            sid = parse_int_id(getattr(t, "sid", 0))
            tree_id = parse_int_id(getattr(t, "id", 0))

            if sid is not None and tree_id is not None:
                live.add(("yp", sid, tree_id))

    if ms.buffers is not None:
        ut = ms.buffers.saplings
        for layer in range(10):
            for spe_name, _ in UT_SPECIES_FIELDS:
                s = getattr(ut[0][layer], spe_name)
                for cat_code, _ in UT_CATEGORIES:
                    stems = float(getattr(s, f"f_{cat_code}", 0.0) or 0.0)
                    if stems <= 0.0:
                        continue
                    osid = parse_int_id(getattr(s, f"osid_{cat_code}", 0))
                    if osid is not None:
                        live.add(("ut", osid, None))

    return live


def prune_reference_trees_not_in_motti(stand: ForestStand) -> None:

    rt = stand.reference_trees
    if rt is None or rt.size == 0:
        return

    live = collect_live_motti_keys(stand)
    delete_idx = []

    for i, value in enumerate(rt.stratum.tolist()):
        sid = parse_int_id(value)
        if sid is None:
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


def prune_promoted_sapling_reference_trees(stand: ForestStand) -> None:
    """
    Delete old sapling RFs if SID exists in YP vector.
    """
    ms = stand.motti_state
    rt = stand.reference_trees

    if ms is None or ms.yp is None or rt is None or rt.size == 0:
        return

    yp_strata: set[int] = set()
    for i in range(int(ms.ntrees)):
        t = ms.yp[0][i]
        sid = parse_int_id(getattr(t, "sid", 0))
        if sid is not None:
            yp_strata.add(sid)

    if not yp_strata:
        return

    delete_idx: list[int] = []
    for i in range(rt.size):
        if not bool(rt.sapling[i]):
            continue

        sid = parse_int_id(rt.stratum[i])
        if sid is not None and sid in yp_strata:
            delete_idx.append(i)

    if delete_idx:
        rt.delete(np.array(delete_idx, dtype=int))
