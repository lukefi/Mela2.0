from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict, Any, cast
import os
from contextlib import contextmanager

from cffi import FFI

from lukefi.metsi.data.motti.motti_types import (
    FloatPtr,
    IntPtr,
    Motti4Ctrl,
    Motti4FerArray,
    Motti4KorArray,
    Motti4Saplings,
    Motti4Site,
    Motti4Strata,
    Motti4Trees,
    Motti4VcrArray
)


@dataclass
class MottiStateBuffers:
    """Persistent Motti model state buffers that must be carried across Growth calls."""
    saplings: Motti4Saplings        # "Motti4Saplings *"   (ut)
    kor_state: Motti4KorArray       # "Motti4KorArray *"   (kor)
    vcr_state: Motti4VcrArray       # "Motti4VcrArray *"   (vcr)
    apv_state: Motti4KorArray       # "Motti4KorArray *"   (apv)
    fert_array: Motti4FerArray      # "Motti4FerArray *"   (fer)
    numfer: IntPtr                  # "int *"              (numfer)
    ctrl: Motti4Ctrl                # "Motti4Ctrl *"       (o)


@dataclass
class GrowthDeltas:
    tree_ids: List[int]   # IDs of trees that survived in the DLL after growth
    tree_sids: List[int | None]  # stratum ids
    trees_id: List[float]   # diameter increments (xd)
    trees_ih: List[float]   # height increments (xh)
    trees_if: List[float]   # stems/ha delta (Δf)
    trees_age: List[float]   # final biological age
    trees_age13: List[float]  # final breast-height age


@contextmanager
def _maybe_chdir(tmp_dir: Optional[Path] = None):
    if tmp_dir is None:
        yield
        return
    prev = Path.cwd()
    try:
        os.chdir(str(tmp_dir))
        yield
    finally:
        os.chdir(str(prev))


def _resolve_dir_or_file(path_like: Optional[str | Path]) -> Path:
    """
    Turn a user-provided path into an absolute Path. If None, use default.
    """
    if path_like is None:
        return _default_data_dir()
    p = Path(os.path.expanduser(os.path.expandvars(str(path_like))))
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def _default_data_dir() -> Path:
    """
    Resolve default data_dir as {repository_root}/data/motti,
    with optional override via MOTTI_DATA_DIR.
    """
    env = os.environ.get("MOTTI_DATA_DIR")
    if env:
        return Path(os.path.expanduser(os.path.expandvars(env))).resolve()
    repo = _find_repo_root(Path.cwd())
    base = repo if repo else Path.cwd()
    return (base / "data" / "motti").resolve()


def _find_repo_root(start: Path) -> Optional[Path]:
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


def _resolve_shared_object(p: str | Path) -> Path:
    """
    Resolve a Motti shared library inside a directory, or pass through an exact file path.
    Raises ValueError if p is None. Returns a Path (may be a directory if nothing matched).
    """

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


class Motti4DLL:
    data_dir: Path
    ffi: FFI
    lib: Any

    @classmethod
    def load(cls, data_dir: str | Path | None = None):

        resolved_dir = _resolve_dir_or_file(data_dir)
        so_path = _resolve_shared_object(resolved_dir)

        lib_path = Path(so_path).resolve()
        cls.data_dir = Path(resolved_dir)
        ffi = FFI()
        ffi.cdef(cls._cdef_source())
        # Add DLL search dirs once; keep handles alive

        if hasattr(os, "add_dll_directory"):
            for p in (lib_path.parent, cls.data_dir):
                if p:
                    ps = str(Path(p).resolve())
                    os.add_dll_directory(ps)

        lib = ffi.dlopen(str(lib_path))
        cls.ffi, cls.lib = ffi, lib

    # ---------- helpers ----------
    @classmethod
    def _convert_site_index(cls, mty: int | float) -> int:
        # Prefer DLL helper; otherwise cap <= 6 (matches their Convert_Site policy)
        if hasattr(cls.lib, "Convert_Site"):
            return int(round(float(cls.lib.Convert_Site(float(mty)))))

        return min(int(mty), 6)

    # ---------- FFI ----------
    @classmethod
    def _cdef_source(cls) -> str:
        with open(cls.data_dir.joinpath("motti.h"), "r", encoding="utf-8") as header:
            return header.read()

    # ---------- site + trees ----------
    @classmethod
    def new_site(
        cls,
        *,
        Y: float, X: float, Z: float = -1.0,
        lake: float = 0.0, sea: float = 0.0,
        mal: int = 1, mty: int = 3, verl: int = 2, verlt: int = 0,
        xt_regen: int = 1, xt_muok: int = 1, xt_raiv: int = 1, sid: int = 1,
        fthin: bool = False, xt_thin: int = 1, xt_fert: int = 1,
        xt_thoit: int = 1, drain: int = 1, xt_ndrain: int = 1,
        alr: int = 1,
        year: Optional[float] = 2010.0,   # safe default if caller does not provide
        step: float = 5.0,
        convert_mela_site: bool = True,
        spedom: Optional[int] = None,
        spedom2: Optional[int] = None,
        nstorey: float = 1.0,
        gstorey: float = 1.0,
    ):
        """
        IMPORTANT: Matches C flow -> SiteInit first, then fill fields (no dd), then CheckYY.
        If Z is unknown, pass Z=-1.0 to let the DLL infer it.
        """
        ffi, lib = cls.ffi, cls.lib
        yy = cast(Motti4Site, ffi.new("Motti4Site *"))

        # SiteInit with only Y,X,Z
        rv = cast(IntPtr, ffi.new("int *"))
        with _maybe_chdir(cls.data_dir):
            lib.Motti4SiteInit(yy,
                               ffi.new("float *", Y),
                               ffi.new("float *", X),
                               ffi.new("float *", Z),
                               rv)
        if rv[0] != 0:
            raise RuntimeError(f"Motti4SiteInit failed (rv={rv[0]})")

        yy.Y = Y
        yy.X = X
        yy.Z = Z
        yy.lake = lake
        yy.sea = sea
        yy.mal = mal
        yy.mty = cls._convert_site_index(mty) if convert_mela_site else mty
        yy.verl = verl
        yy.verlt = verlt
        yy.xt_regen = xt_regen
        yy.xt_muok = xt_muok
        yy.xt_raiv = xt_raiv
        yy.sid = sid

        yy.fthin = fthin
        yy.xt_thin = xt_thin
        yy.xt_fert = xt_fert
        yy.xt_thoit = xt_thoit
        yy.drain = drain
        yy.xt_ndrain = xt_ndrain

        yy.alr = alr
        if year is not None:
            yy.year = year
        yy.step = step
        yy.nstorey = 1.0
        yy.gstorey = 1.0

        yy.nstorey = nstorey
        yy.gstorey = gstorey
        if spedom is not None:
            yy.spedom = spedom
        if spedom2 is not None:
            yy.spedom2 = spedom2

        # 3) Validate
        nerr = cast(IntPtr, ffi.new("int *"))
        err = cast(IntPtr, ffi.new("int *"))
        with _maybe_chdir(cls.data_dir):
            lib.Motti4CheckYY(yy, nerr, err)
        if nerr[0] != 0:
            raise RuntimeError(f"Motti4CheckYY signaled problem (nerr={nerr[0]}, err={err[0]})")

        return yy

    @classmethod
    def new_trees(cls, trees_py: list[dict]) -> Tuple[Motti4Trees, int]:
        """
            fields used: id, sid, f, d13, h, spe, age, age13, cr, snt
        """
        ypp = cast(Motti4Trees, cls.ffi.new("Motti4Trees *"))
        yp = ypp[0]
        for i, t in enumerate(trees_py):
            yp[i].id = int(t.get("id", i + 1))
            yp[i].sid = float(t.get("sid", 0))
            yp[i].f = float(t.get("f", 0.0))
            yp[i].d13 = float(t.get("d13", 0.0))
            yp[i].h = float(t.get("h", 0.0))
            yp[i].spe = float(t.get("spe", 1))
            yp[i].age = float(t.get("age", 0.0))
            yp[i].age13 = float(t.get("age13", 0.0))
            yp[i].cr = float(t.get("cr", 0.0))
            yp[i].snt = float(t.get("snt", 1))
            yp[i].crerror = 0.0  # clear before growth
            yp[i].storie = float(t.get("storie", 2.0))
        return ypp, len(trees_py)

    @classmethod
    def new_strata(cls, strata_py: list[dict]) -> Motti4Strata:
        """
        Builds Motti4Strata from FDM strata.
        """
        yo = cast(Motti4Strata, cls.ffi.new("Motti4Strata *"))

        max_n = min(len(strata_py), 10)
        for i in range(max_n):
            s = strata_py[i]
            yo[0][i].spe = float(s.get("spe", 0.0))
            yo[0][i].age = float(s.get("age", 0.0))
            yo[0][i].ba = float(s.get("ba", 0.0))
            yo[0][i].f = float(s.get("f", 0.0))
            yo[0][i].h = float(s.get("h", 0.0))
            yo[0][i].hw = float(s.get("hw", 0.0))
            yo[0][i].d = float(s.get("d", 0.0))
            yo[0][i].dg = float(s.get("dg", 0.0))
            yo[0][i].storey = float(s.get("storey", 0.0))
            yo[0][i].st = float(s.get("st", 0.0))
            yo[0][i].sid = float(s.get("sid", 0.0))

        return yo
    # ---------- persistent state buffers ----------

    @classmethod
    def alloc_state_buffers(cls, ctrl: Optional[dict] = None) -> MottiStateBuffers:
        """Allocate persistent buffers that must be reused across Growth calls."""
        ffi = cls.ffi
        saplings = cast(Motti4Saplings, ffi.new("Motti4Saplings *"))
        kor_state = cast(Motti4KorArray, ffi.new("Motti4KorArray *"))
        vcr_state = cast(Motti4VcrArray, ffi.new("Motti4VcrArray *"))
        apv_state = cast(Motti4KorArray, ffi.new("Motti4KorArray *"))
        fert_array = cast(Motti4FerArray, ffi.new("Motti4FerArray *"))
        numfer = cast(IntPtr, ffi.new("int *", 0))
        motti_control = cast(Motti4Ctrl, ffi.new("Motti4Ctrl *"))
        motti_control.death_tree = 1
        if ctrl:
            if "death_tree" in ctrl:
                motti_control.death_tree = int(bool(ctrl["death_tree"]))
            if "death_forest" in ctrl:
                motti_control.death_forest = int(bool(ctrl["death_forest"]))
            if "calibrate" in ctrl:
                motti_control.calibrate = int(bool(ctrl["calibrate"]))
        return MottiStateBuffers(
            saplings=saplings,
            kor_state=kor_state,
            vcr_state=vcr_state,
            apv_state=apv_state,
            fert_array=fert_array,
            numfer=numfer,
            ctrl=motti_control,
        )

    @classmethod
    def clone_state_buffers(cls, buffers: MottiStateBuffers) -> MottiStateBuffers:
        """Deep-copy buffers for branching."""
        ffi = cls.ffi
        out = cls.alloc_state_buffers(ctrl={
            "death_tree": int(bool(buffers.ctrl.death_tree)),
            "death_forest": int(bool(buffers.ctrl.death_forest)),
            "calibrate": int(bool(buffers.ctrl.calibrate)),
        })
        ffi.memmove(cast(FFI.CData, out.saplings), cast(FFI.CData, buffers.saplings), ffi.sizeof("Motti4Saplings"))
        ffi.memmove(cast(FFI.CData, out.kor_state), cast(FFI.CData, buffers.kor_state), ffi.sizeof("Motti4KorArray"))
        ffi.memmove(cast(FFI.CData, out.vcr_state), cast(FFI.CData, buffers.vcr_state), ffi.sizeof("Motti4VcrArray"))
        ffi.memmove(cast(FFI.CData, out.apv_state), cast(FFI.CData, buffers.apv_state), ffi.sizeof("Motti4KorArray"))
        ffi.memmove(cast(FFI.CData, out.fert_array), cast(FFI.CData, buffers.fert_array), ffi.sizeof("Motti4FerArray"))
        out.numfer[0] = int(buffers.numfer[0])
        return out

    @classmethod
    def clone_site(cls, yy: Motti4Site) -> Motti4Site:
        """Deep-copy a site struct (yy) for branching."""
        ffi = cls.ffi
        yy2 = cast(Motti4Site, ffi.new("Motti4Site *"))
        ffi.memmove(cast(FFI.CData, yy2), cast(FFI.CData, yy), ffi.sizeof("Motti4Site"))
        return yy2

    @classmethod
    def clone_trees(cls, yp: Motti4Trees) -> Motti4Trees:
        """Deep-copy a full Motti4Trees buffer (fixed 1000-tree array)."""
        ffi = cls.ffi
        yp2 = cast(Motti4Trees, ffi.new("Motti4Trees *"))
        ffi.memmove(cast(FFI.CData, yp2), cast(FFI.CData, yp), ffi.sizeof("Motti4Trees"))
        return yp2

    @classmethod
    def grow_with_state(
        cls,
        yy: Motti4Site,
        yp: Motti4Trees,
        numtrees: int,
        buffers: MottiStateBuffers,
        step: int = 5,
    ) -> GrowthDeltas:
        """
        Growth using persistent buffers that are carried across calls.

        step=0 performs a single zero-step Growth call, which is used to refresh
        derived fields after Python-side yp edits.
        """
        ffi, lib = cls.ffi, cls.lib

        ntrees_p = cast(IntPtr, ffi.new("int *", int(numtrees)))
        rv = ffi.new("int *")

        acc_id: Dict[int, float] = {}
        acc_ih: Dict[int, float] = {}
        acc_if: Dict[int, float] = {}
        prev_f: Dict[int, float] = {int(yp[0][i].id): float(yp[0][i].f) for i in range(ntrees_p[0])}

        remaining = int(step)
        runs_left = 1 if remaining <= 0 else None
        while remaining > 0 or runs_left:
            yy._290 = 0.0  # pylint: disable=protected-access
            for i in range(ntrees_p[0]):
                yp[0][i].crerror = 0.0

            current_step = remaining if remaining > 0 else 0
            step_p = cast(IntPtr, ffi.new("int *", current_step))
            rv[0] = 0
            with _maybe_chdir(cls.data_dir):
                lib.Motti4Growth(
                    yy, yp,
                    buffers.saplings,
                    buffers.kor_state,
                    buffers.vcr_state,
                    buffers.apv_state,
                    ntrees_p,
                    buffers.fert_array,
                    buffers.numfer,
                    buffers.ctrl,
                    step_p,
                    rv
                )
            if rv[0] != 0:
                raise RuntimeError(f"Motti4Growth failed (rv={rv[0]})")

            for i in range(ntrees_p[0]):
                tid = int(yp[0][i].id)
                acc_id[tid] = acc_id.get(tid, 0.0) + float(yp[0][i].xd)
                acc_ih[tid] = acc_ih.get(tid, 0.0) + float(yp[0][i].xh)
                nf = float(yp[0][i].f)
                pf = prev_f.get(tid, nf)
                acc_if[tid] = acc_if.get(tid, 0.0) + (nf - pf)
                prev_f[tid] = nf

            if runs_left:
                runs_left -= 1
                continue

            done = int(step_p[0])
            if done <= 0:
                break
            remaining -= done

        ids_now = [int(yp[0][i].id) for i in range(ntrees_p[0])]
        sids_now = []
        for i in range(ntrees_p[0]):
            raw_sid = yp[0][i].sid
            sid = int(raw_sid)
            sids_now.append(sid if sid > 0 else None)
        out_id = [acc_id.get(tid, 0.0) for tid in ids_now]
        out_ih = [acc_ih.get(tid, 0.0) for tid in ids_now]
        out_if = [acc_if.get(tid, 0.0) for tid in ids_now]
        out_age = [float(yp[0][i].age) for i in range(ntrees_p[0])]
        out_age13 = [float(yp[0][i].age13) for i in range(ntrees_p[0])]

        return GrowthDeltas(
            tree_ids=ids_now,
            tree_sids=sids_now,
            trees_id=out_id,
            trees_ih=out_ih,
            trees_if=out_if,
            trees_age=out_age,
            trees_age13=out_age13,
        )

    @classmethod
    def update_after_import(
        cls,
        yy: Motti4Site,
        yp: Motti4Trees,
        numtrees: int,
        buffers: MottiStateBuffers,
    ) -> int:
        """
        Called after Motti4InitVer2
        """
        ffi, lib = cls.ffi, cls.lib
        ntrees_p = cast(IntPtr, ffi.new("int *", int(numtrees)))
        rv = cast(IntPtr, ffi.new("int *"))

        with _maybe_chdir(cls.data_dir):
            lib.Motti4UpdateAfterImport(
                yy,
                yp,
                buffers.saplings,
                buffers.kor_state,
                buffers.vcr_state,
                buffers.apv_state,
                ntrees_p,
                rv,
            )
        if rv[0] != 0:
            raise RuntimeError(f"Motti4UpdateAfterImport failed (rv={rv[0]})")

        return int(ntrees_p[0])

    @classmethod
    def regenerate_with_state(
        cls,
        yy: Motti4Site,
        yp: Motti4Trees,
        numtrees: int,
        buffers: MottiStateBuffers,
        *,
        method: list[float],
        step: int = 0,
    ) -> int:
        """
        Call Motti4Regenerate against persistent state buffers.

        method:
        [0] regeneration method (1 natural, 2 sowing, 3 planting)
        [1] survival percent [0..100]
        [2] cultivated tree species
        [3] amount (pcs/ha)
        [4] soil preparation type
        [5] clearing (0/1)
        [6] seed tree species
        [7..9] unused, kept as 0.0
        """
        ffi, lib = cls.ffi, cls.lib

        if len(method) > 10:
            raise ValueError("Motti4Regenerate method vector may contain at most 10 values")

        method_vec = method + [0.0] * (10 - len(method))
        method_p = cast(list[float], ffi.new("float[10]", method_vec))

        ntrees_p = cast(IntPtr, ffi.new("int *", int(numtrees)))
        step_p = cast(IntPtr, ffi.new("int *", int(step)))
        rv = cast(IntPtr, ffi.new("int *"))

        with _maybe_chdir(cls.data_dir):
            lib.Motti4Regenerate(
                method_p,
                yy,
                yp,
                buffers.saplings,
                buffers.kor_state,
                buffers.vcr_state,
                buffers.apv_state,
                ntrees_p,
                step_p,
                rv,
            )

        if rv[0] != 0:
            raise RuntimeError(f"Motti4Regenerate failed (rv={rv[0]})")

        return int(ntrees_p[0])

    @staticmethod
    def _normalize_remaining_n_array(remaining_n: list[int]) -> list[int]:
        """
        Normalize remainingN into a 10-slot int list where indices 1..9 are species.
        Index 0 is kept as 0 because Motti species slots are documented as 1..9.
        Accepts:
          - list/tuple/ndarray of length 9  -> mapped to slots 1..9
          - list/tuple/ndarray of length 10 -> used as-is
          - dict {species_slot: stems}
        """

        vals = [int(x) for x in remaining_n]

        if len(vals) == 9:
            return [0] + [max(v, 0) for v in vals]

        if len(vals) == 10:
            out = [max(v, 0) for v in vals]
            out[0] = 0
            return out

        raise ValueError(
            "remaining_n must be dict or an array of length 9 (species 1..9) "
            "or length 10 (slot 0 unused, species in 1..9)"
        )

    @classmethod
    def pct_guidelines_with_state(
        cls,
        yy: Motti4Site,
        yp: Motti4Trees,
        numtrees: int,
        buffers: MottiStateBuffers,
    ) -> list[int]:
        """
        Call Motti4PCTGuidelines against persistent state buffers and return
        a 10-slot list where indices 1..9 correspond to remaining stem count for each species.
        """
        ffi, lib = cls.ffi, cls.lib

        ntrees_p = cast(IntPtr, ffi.new("int *", int(numtrees)))
        remaining_n_p = cast(list[int], ffi.new("int[10]", [0] * 10))
        rv = cast(IntPtr, ffi.new("int *"))

        with _maybe_chdir(cls.data_dir):
            lib.Motti4PCTGuidelines(
                yy,
                yp,
                buffers.saplings,
                buffers.kor_state,
                buffers.vcr_state,
                buffers.apv_state,
                ntrees_p,
                remaining_n_p,
                rv,
            )

        if rv[0] != 0:
            raise RuntimeError(f"Motti4PCTGuidelines failed (rv={rv[0]})")

        return [int(remaining_n_p[i]) for i in range(10)]

    @classmethod
    def earlycare_with_state(
        cls,
        yy: Motti4Site,
        yp: Motti4Trees,
        numtrees: int,
        buffers: MottiStateBuffers,
        *,
        imode: int = 0,
    ) -> list[float]:
        """
        Call Motti4EarlyCare against persistent state buffers.

        Returns info array where:
          info[0] = remaining stem count
          info[1] = removed stem count
          info[2] = removed d13 (cm)
          info[3] = removed height (m)
        """
        ffi, lib = cls.ffi, cls.lib

        if imode not in (0, 1):
            raise ValueError("imode must be 0 or 1")

        ntrees_p = cast(IntPtr, ffi.new("int *", int(numtrees)))
        info_p = cast(list[float], ffi.new("float[10]", [0.0] * 10))
        imode_p = cast(IntPtr, ffi.new("int *", int(imode)))
        rv = cast(IntPtr, ffi.new("int *"))

        with _maybe_chdir(cls.data_dir):
            lib.Motti4EarlyCare(
                yy,
                yp,
                buffers.saplings,
                buffers.kor_state,
                buffers.vcr_state,
                buffers.apv_state,
                ntrees_p,
                info_p,
                imode_p,
                rv,
            )

        if rv[0] != 0:
            raise RuntimeError(f"Motti4EarlyCare failed (rv={rv[0]})")

        return [float(info_p[i]) for i in range(10)]

    @classmethod
    def fillin_planting_with_state(
        cls,
        yy: Motti4Site,
        yp: Motti4Trees,
        numtrees: int,
        buffers: MottiStateBuffers,
        *,
        rspe: int,
        num: float,
        osite_id: int,
    ) -> int:
        """
        Call Motti4FillinPlanting against persistent state buffers.

        Parameters
        ----------
        rspe : int
            Motti species slot for planted species.
        num : float
            Number of planted saplings per hectare.
        osite_id : int
            Identifier for the planted sapling cohort / stratum.
        """
        ffi, lib = cls.ffi, cls.lib

        ntrees_p = cast(IntPtr, ffi.new("int *", int(numtrees)))
        rspe_p = cast(IntPtr, ffi.new("int *", int(rspe)))
        num_p = cast(FloatPtr, ffi.new("float *", float(num)))
        osite_id_p = cast(IntPtr, ffi.new("int *", int(osite_id)))
        rv = cast(IntPtr, ffi.new("int *"))

        with _maybe_chdir(cls.data_dir):
            lib.Motti4FillinPlanting(
                yy,
                yp,
                buffers.saplings,
                buffers.kor_state,
                buffers.vcr_state,
                buffers.apv_state,
                ntrees_p,
                rspe_p,
                num_p,
                osite_id_p,
                rv,
            )

        if rv[0] != 0:
            raise RuntimeError(f"Motti4FillinPlanting failed (rv={rv[0]})")

        return int(ntrees_p[0])

    @classmethod
    def after_seedtree_cutting_with_state(
        cls,
        yy: Motti4Site,
        yp: Motti4Trees,
        numtrees: int,
        buffers: MottiStateBuffers,
    ) -> int:
        """
        Call Motti4AfterSeedtreeCutting after a seed-tree cutting.

        Remaining seed trees must already have tree class / puuluokka = 3
        in the YP vector before this function is called. If Motti creates
        natural regeneration, it is written to the persistent sapling buffer.
        """
        ffi, lib = cls.ffi, cls.lib

        ntrees_p = cast(IntPtr, ffi.new("int *", int(numtrees)))
        ierror = cast(IntPtr, ffi.new("int *"))
        rv = cast(IntPtr, ffi.new("int *"))

        with _maybe_chdir(cls.data_dir):
            lib.Motti4AfterSeedtreeCutting(
                yy,
                yp,
                buffers.saplings,
                buffers.kor_state,
                buffers.vcr_state,
                buffers.apv_state,
                ntrees_p,
                ierror,
                rv,
            )

        if rv[0] != 0:
            raise RuntimeError(f"Motti4AfterSeedtreeCutting failed (rv={rv[0]})")

        return int(ntrees_p[0])

    @classmethod
    def seedling_delay_with_state(
        cls,
        yy: Motti4Site,
        buffers: MottiStateBuffers,
        *,
        istep: int,
    ) -> None:
        """
        Call Motti4SeedingAgeShift against persistent state buffers.

        Only the last sapling layer is affected by Motti, and only saplings with
        age 0 or 1 years are adjusted. Positive istep increases age, negative
        istep decreases age.
        """
        ffi, lib = cls.ffi, cls.lib

        rv = cast(IntPtr, ffi.new("int *"))
        istep_p = cast(IntPtr, ffi.new("int *", int(istep)))

        with _maybe_chdir(cls.data_dir):
            lib.Motti4SeedingAgeShift(
                yy,
                buffers.saplings,
                istep_p,
                rv,
            )

        if rv[0] != 0:
            raise RuntimeError(f"Motti4SeedingAgeShift failed (rv={rv[0]})")

    @classmethod
    def mineral_soils_fertilization_with_state(
        cls,
        yy: Motti4Site,
        yp: Motti4Trees,
        numtrees: int,
        buffers: MottiStateBuffers,
        *,
        ftype: int,
        amount_n: float,
        bool_phosphorus: int = 0,
    ) -> list[float]:
        """
        Call Motti4MineralSoilsFertilization against persistent state buffers.

        Parameters
        ----------
        ftype
            Fertilization type code passed through to Motti.
        amount_n
            Nitrogen amount.
        bool_phosphorus
            0/1 flag for phosphorus fertilization.

        Returns
        -------
        list[float]
            Raw 10-slot response vector returned by Motti.
        """
        ffi, lib = cls.ffi, cls.lib

        ntrees_p = cast(IntPtr, ffi.new("int *", int(numtrees)))
        ftype_p = cast(IntPtr, ffi.new("float *", float(ftype)))
        amount_n_p = cast(FloatPtr, ffi.new("float *", float(amount_n)))
        bool_phosphorus_p = cast(IntPtr, ffi.new("int *", int(bool(bool_phosphorus))))
        response_p = cast(list[float], ffi.new("float[10]", [0.0] * 10))

        with _maybe_chdir(cls.data_dir):
            lib.Motti4MineralSoilsFertilization(
                ftype_p,
                amount_n_p,
                bool_phosphorus_p,
                yy,
                yp,
                buffers.saplings,
                buffers.kor_state,
                buffers.vcr_state,
                buffers.apv_state,
                ntrees_p,
                buffers.fert_array,
                buffers.numfer,
                response_p,
            )

        return [float(response_p[i]) for i in range(10)]

    @classmethod
    def pct_with_state(
        cls,
        yy: Motti4Site,
        yp: Any,
        numtrees: int,
        buffers: MottiStateBuffers,
        *,
        remaining_n: list[int],
    ) -> int:
        """
        Call Motti4PCT against persistent state buffers.

        remaining_n must describe species-wise remaining stem counts for
        species slots 1..9. Accepted forms:
          - dict {species_slot: stems}
          - list/tuple length 9
          - list/tuple length 10 (slot 0 unused)
        """
        ffi, lib = cls.ffi, cls.lib

        remaining_arr = cls._normalize_remaining_n_array(remaining_n)

        ntrees_p = cast(IntPtr, ffi.new("int *", int(numtrees)))
        remaining_n_p = cast(list[int], ffi.new("int[10]", remaining_arr))
        rv = cast(IntPtr, ffi.new("int *"))

        with _maybe_chdir(cls.data_dir):
            lib.Motti4PCT(
                yy,
                yp,
                buffers.saplings,
                buffers.kor_state,
                buffers.vcr_state,
                buffers.apv_state,
                ntrees_p,
                remaining_n_p,
                rv,
            )

        if rv[0] != 0:
            raise RuntimeError(f"Motti4PCT failed (rv={rv[0]})")

        return int(ntrees_p[0])

    @classmethod
    def initialize_with_state(cls,
                              yo: Motti4Strata,
                              yy: Motti4Site,
                              yp: Motti4Trees,
                              numtrees: int,
                              buffers: MottiStateBuffers) -> int:
        ffi, lib = cls.ffi, cls.lib

        ntrees_p = cast(IntPtr, ffi.new("int *", int(numtrees)))
        err = cast(IntPtr, ffi.new("int *"))
        rv = cast(IntPtr, ffi.new("int *"))

        with _maybe_chdir(cls.data_dir):
            lib.Motti4InitVer2(
                yo,
                yy,
                buffers.saplings,
                buffers.kor_state,
                buffers.vcr_state,
                buffers.apv_state,
                yp,
                buffers.ctrl,
                ntrees_p,
                err,
                rv,
            )

        if rv[0] != 0 or err[0] != 0:
            raise RuntimeError(f"Motti4InitVer2 failed (rv={rv[0]}, err={err[0]})")

        return cls.update_after_import(yy, yp, int(ntrees_p[0]), buffers)
