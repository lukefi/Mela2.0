import os
from typing import Any, Optional, Dict, Union, Iterable
from pathlib import Path
import numpy as np
from lukefi.metsi.domain.natural_processes.motti_dll_wrapper import (
    Motti4DLL,
    GrowthDeltas,
)
from lukefi.metsi.data.enums.internal import (
    LandUseCategory,
    TreeSpecies,
    Storey,
    CONIFEROUS_SPECIES,
    DECIDUOUS_SPECIES
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
    find_sapling_reference_tree_index


)
from lukefi.metsi.domain.natural_processes.natural_process_wrapper import natural_process_transition
from lukefi.metsi.sim.collected_data import OpTuple


def debug_dump_reference_trees(stand: ForestStand, label: str) -> None:
    rt = getattr(stand, "reference_trees", None)
    print(f"\n=== {label} / reference_trees ===")
    if rt is None:
        print("reference_trees = None")
        return
    print("rt.size =", rt.size)
    for i in range(rt.size):
        print(
            "  RT",
            i,
            "sid=", rt.stratum[i],
            "tree_number=", rt.tree_number[i],
            "sapling=", bool(rt.sapling[i]),
            "f=", float(rt.stems_per_ha[i]),
            "d=", float(rt.breast_height_diameter[i]),
            "h=", float(rt.height[i]),
            "age=", float(rt.biological_age[i]),
            "age13=", float(rt.breast_height_age[i]),
            "ba=", float(rt.basal_area[i]),
            "vol=", float(rt.volume[i]),
        )


def debug_dump_motti_state_raw(ms: MottiState, label: str) -> None:
    print(f"\n=== {label} ===")
    if ms is None:
        print("ms = None")
        return

    print("ms.ntrees =", ms.ntrees)

    # Dump yp rows that Motti says are active
    if ms.yp is None:
        print("yp = None")
    else:
        print("yp active rows:")
        for i in range(int(ms.ntrees)):
            t = ms.yp[0][i]
            print(
                "  YP",
                i,
                "sid=", getattr(t, "sid", None),
                "id=", getattr(t, "id", None),
                "spe=", getattr(t, "spe", None),
                "f=", getattr(t, "f", None),
                "d13=", getattr(t, "d13", None),
                "h=", getattr(t, "h", None),
                "age=", getattr(t, "age", None),
                "age13=", getattr(t, "age13", None),
            )

    # Dump ut rows with positive stems
    if ms.buffers is None:
        print("buffers = None")
    else:
        ut = ms.buffers.saplings
        found_ut = False
        for layer in range(10):
            for spe_name, _internal_species in UT_SPECIES_FIELDS:
                s = getattr(ut[0][layer], spe_name)
                year = getattr(s, "year", None)
                for cat_code, _cat_label in UT_CATEGORIES:
                    stems = float(getattr(s, f"f_{cat_code}", 0.0) or 0.0)
                    if stems > 0.0:
                        found_ut = True
                        print(
                            "  UT",
                            "layer=", layer,
                            "spe=", spe_name,
                            "cat=", cat_code,
                            "year=", year,
                            "osid=", getattr(s, f"osid_{cat_code}", None),
                            "f=", stems,
                            "h=", getattr(s, f"h_{cat_code}", None),
                            "d=", getattr(s, f"d_{cat_code}", None),
                            "age=", getattr(s, f"age_{cat_code}", None),
                            "age13=", getattr(s, f"age13_{cat_code}", None),
                        )
        if not found_ut:
            print("ut positive rows: none")


def _normalize_ut_years_for_fdm(stand: ForestStand) -> None:

    if stand.motti_state is None or stand.motti_state.buffers is None:
        return

    ut = stand.motti_state.buffers.saplings
    for layer in range(10):
        for spe_name, _internal_species in UT_SPECIES_FIELDS:
            s = getattr(ut[0][layer], spe_name)
            try:
                raw_year = float(s.year)
            except (TypeError, ValueError):
                continue

            if raw_year == -1.0:
                continue

            s.year = stand.start_year + raw_year


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


def _grouped_yp_indices_by_sid(ms: MottiState) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for i in range(int(ms.ntrees)):
        sid = parse_int_id(getattr(ms.yp[0][i], "sid", 0))
        if sid is None:
            continue
        grouped.setdefault(sid, []).append(i)

    for sid, indices in grouped.items():
        indices.sort(key=lambda j: int(getattr(ms.yp[0][j], "id", 0) or 0))
    return grouped


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

                print(
                    "UT ROW",
                    "osid=", osid,
                    "stems=", stems,
                    "sapling_idx=", find_sapling_reference_tree_index(rt, osid),
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
            "tree_category": "1",
            "management_category": 1,
            "storey": int(storey),
            "basal_area": float(t.ba) if getattr(t, "ba", None) is not None else 0.0,
            "volume": float(t.vol) if getattr(t, "vol", None) is not None else 0.0,
        }

        if idx is None:
            rt.create(row)
        else:
            rt.update(row, idx)


def prune_reference_trees_not_in_yp(stand: ForestStand) -> None:
    """
    Keep only ReferenceTrees that have a live match in the current Motti YP vector.

    This is intentionally stricter than prune_reference_trees_not_in_motti():
    after Motti init / refresh we first rebuild the tree layer from YP, then we
    rebuild the sapling layer from UT. Any pre-existing RF that is not present in
    YP is deleted before UT rows are recreated.
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


def reconcile_reference_trees_from_motti(stand: ForestStand) -> None:
    """Rebuild RFs from Motti in the intended order: YP first, then UT."""
    sync_yp_to_reference_trees(stand)
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


def _build_motti_strata_py(stand: ForestStand) -> list[dict]:
    """
    Convert stand.tree_strata into Python dicts for Motti4Strata.

    Uncertain fields:
      hw -> temporary fallback to mean_height
      dg -> temporary fallback to mean_diameter
      st -> temporary dummy 0.0
    """
    strata = getattr(stand, "tree_strata", None)
    if strata is None or strata.size == 0:
        return []

    out: list[dict] = []

    for i in range(min(strata.size, 10)):
        spe_raw = int(strata.species[i])
        if spe_raw < 0:
            continue

        biological_age = float(np.nan_to_num(strata.biological_age[i], nan=0.0))
        basal_area = float(np.nan_to_num(strata.basal_area[i], nan=0.0))
        stems_main = float(np.nan_to_num(strata.stems_per_ha[i], nan=0.0))
        # stems_sap = float(np.nan_to_num(strata.sapling_stems_per_ha[i], nan=0.0))
        mean_height = float(np.nan_to_num(strata.mean_height[i], nan=0.0))
        mean_diameter = float(np.nan_to_num(strata.mean_diameter[i], nan=0.0))
        origin = safe_storey_value(strata.origin[i])
        storey = safe_storey_value(strata.storey[i])

        stratum_sid = parse_int_id(strata.stratum_number[i])
        if stratum_sid is None:
            stratum_sid = i + 1

        spe = float(species_to_motti(spe_raw))
        out.append({
            "spe": spe,
            "age": biological_age,
            "ba": basal_area,
            "f": stems_main,
            "h": mean_height,
            "hw": mean_height,      # ppa-weighted height
            "d": mean_diameter,
            "dg": mean_diameter,    # ppa-weighted keskiläpimitta
            "storey": storey,
            "st": origin,
            "sid": float(stratum_sid),
        })

    return out


def _spedom(rt: ReferenceTrees | Any | None) -> int:
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
    spe_codes = np.asarray([species_to_motti(int(s)) for s in rt.species.tolist()], dtype=int)

    # Basal area per tree: stems_per_ha * π * (0.5 * d_cm * 0.01 m/cm)^2
    d_cm = np.nan_to_num(rt.breast_height_diameter, nan=0.0)
    f_ha = np.nan_to_num(rt.stems_per_ha, nan=0.0)
    ba_per_tree = f_ha * np.pi * (0.5 * d_cm * 0.01) ** 2  # m²/ha contribution

    # Sum BA per species code
    per: Dict[int, float] = {}
    for code, ba in zip(spe_codes.tolist(), ba_per_tree.tolist()):
        per[code] = per.get(code, 0.0) + float(ba)

    use_basal = any(v > 0.0 for v in per.values())
    if not use_basal:
        per.clear()
        # Fallback: stems/ha totals per species
        for code, stems in zip(spe_codes.tolist(), f_ha.tolist()):
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

    # Create same-length strata object with default values in every column
    stripped = TreeStrata(size=n)

    # Keeping only fields that should survive
    stripped.identifier = stand.tree_strata.identifier.copy()
    stripped.origin = stand.tree_strata.origin.copy()
    stripped.storey = stand.tree_strata.storey.copy()

    stand.tree_strata = stripped


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

        print("ENSURE_STATE ENTER",
              "has_ms=", getattr(self.stand, "motti_state", None) is not None,
              "rt.size=", self.stand.reference_trees.size,
              "strata.size=", 0 if self.stand.tree_strata is None else self.stand.tree_strata.size)

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
        spe_vec = np.asarray([species_to_motti(int(s)) for s in rt.species.tolist()], dtype=int)

        stratum_ids = [
            parse_int_id(v) or (self.stand.stand_id or (idx + 1))
            for idx, v in enumerate(rt.stratum.tolist())
        ]

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
            }
            for i, sid, f, d, hh, sp, a, a13, c, o in zip(
                ids.tolist(),
                stratum_ids,
                stems.tolist(),
                d13.tolist(),
                h.tolist(),
                spe_vec.tolist(),
                age.tolist(),
                age13.tolist(),
                cr.tolist(),
                origin.astype(int).tolist(),
            )
        ]
        yp, ntrees = self.dll.new_trees(trees_py)

        print("\nINIT INPUT TREES_PY:")
        for row in trees_py:
            print(" ", row)
        yp, ntrees = self.dll.new_trees(trees_py)

        strata_py = _build_motti_strata_py(self.stand)
        print("\nINIT INPUT STRATA_PY:")
        for row in strata_py:
            print(" ", row)
        yo = self.dll.new_strata(strata_py)

        buffers = self.dll.alloc_state_buffers(ctrl=None)

        ntrees = self.dll.initialize_with_state(
            yo=yo,
            yy=yy,
            yp=yp,
            numtrees=int(ntrees),
            buffers=buffers,
        )

        temp_ms = MottiState(
            dll=self.dll,
            yy=yy,
            yp=yp,
            ntrees=int(ntrees),
            buffers=buffers,
            signature=tuple(ids.tolist()),
        )

        debug_dump_motti_state_raw(temp_ms, "after initialize_with_state")
        debug_dump_yy_sapling_storey(temp_ms, "after initialize_with_state")
        # _strip_tree_strata(self.stand)

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

        _normalize_ut_years_for_fdm(self.stand)
        reconcile_reference_trees_from_motti(self.stand)

        return self.stand.motti_state

    def evolve(self, step: int = 5, sim_year: int = 0) -> GrowthDeltas:
        state = self.ensure_state(step=step, sim_year=sim_year)
        if state is None:
            return GrowthDeltas(tree_ids=[], tree_sids=[], trees_id=[], trees_ih=[], trees_if=[],
                                trees_age=[], trees_age13=[]
                                )

        state.yy.year = sim_year - self.stand.start_year
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

def species_to_motti(spe: int) -> int:
    """
    Map internal TreeSpecies -> Motti species codes directly.
    - Keep main species 1..5 as-is
    - Collapse both alders (GREY_ALDER, COMMON_ALDER) to 6
    - If in CONIFEROUS_SPECIES -> 8
    - If in DECIDUOUS_SPECIES -> 9
    """
    ts = TreeSpecies(int(spe))
    if ts in (TreeSpecies.PINE, TreeSpecies.SPRUCE,
              TreeSpecies.SILVER_BIRCH, TreeSpecies.DOWNY_BIRCH,
              TreeSpecies.ASPEN):
        return int(ts)
    if ts in (TreeSpecies.GREY_ALDER, TreeSpecies.COMMON_ALDER):
        return int(TreeSpecies.GREY_ALDER)  # Motti uses a single Alder code (6)
    if ts in CONIFEROUS_SPECIES:
        return int(TreeSpecies.OTHER_CONIFEROUS)  # 8
    if ts in DECIDUOUS_SPECIES:
        return int(TreeSpecies.OTHER_DECIDUOUS)  # 9

    raise ValueError(f"Unsupported tree species code: {int(spe)}")


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
    sim_year: int = stand.year or 0

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

    debug_dump_reference_trees(stand, "before evolve")
    ms_before = getattr(stand, "motti_state", None)
    if ms_before is not None:
        debug_dump_motti_state_raw(ms_before, "before evolve")
        debug_dump_yy_sapling_storey(ms_before, "before evolve")

    growth = pred.evolve(step=step, sim_year=sim_year)

    ms_after = getattr(stand, "motti_state", None)
    if ms_after is not None:
        debug_dump_motti_state_raw(ms_after, "after evolve before sync")
        debug_dump_yy_sapling_storey(ms_after, "after evolve before sync")

    print(
        "GROWTH DELTAS SUMMARY",
        "returned_keys=", sorted(
            (int(sid), int(tid))
            for sid, tid in zip(growth.tree_sids, growth.tree_ids)
            if sid is not None
        ),
        "n_growth_rows=", len(growth.tree_ids),
    )

    # Advance simulation year, but do not mutate tree measurements from Python-side deltas.
    stand.year = (stand.year or 0) + step

    reconcile_reference_trees_from_motti(stand)
    debug_dump_reference_trees(stand, "after full motti sync")

    return stand, []


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


def refresh_reference_trees_from_motti_after_yp_change(stand: ForestStand) -> None:
    """
    Rebuild Motti internal state after yp edits, run grow(step=0) and
    then synchronize ReferenceTrees from yp/ut.
    """
    ms = getattr(stand, "motti_state", None)
    if ms is None or ms.yp is None or ms.buffers is None:
        return

    debug_dump_yy_sapling_storey(ms, "before grow_with_state")
    growth = ms.dll.grow_with_state(
        ms.yy,
        ms.yp,
        int(ms.ntrees),
        ms.buffers,
        step=0,
    )
    ms.ntrees = len(growth.tree_ids)

    debug_dump_yy_sapling_storey(ms, "after grow_with_state")

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


def collect_live_motti_keys(stand: ForestStand) -> set[tuple[str, int, int | None]]:
    live: set[tuple[str, int, int | None]] = set()

    ms = getattr(stand, "motti_state", None)
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


def debug_dump_yy_sapling_storey(ms: MottiState, label: str) -> None:
    print(f"\n=== {label} / yy ===")
    if ms is None or ms.yy is None:
        print("yy = None")
        return

    yy = ms.yy

    print("yy.year =", float(yy.year))      # index 50
    print("yy.step =", float(yy.step))
    print("yy.nstorey =", float(yy.nstorey))
    print("yy.gstorey =", float(yy.gstorey))

    print("yy.st2.age100 =", float(yy.st2.age100))  # index 86
    print("yy.st2.h100   =", float(yy.st2.h100))    # 87
    print("yy.st2.g      =", float(yy.st2.g))       # 88
    print("yy.st2.f      =", float(yy.st2.f))       # 89
    print("yy.st2.dg     =", float(yy.st2.dg))      # 90
    print("yy.st2.spe    =", float(yy.st2.spe))     # 91
