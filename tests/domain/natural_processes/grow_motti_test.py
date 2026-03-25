import os
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List
import numpy as np
from lukefi.metsi.domain.natural_processes.motti_dll_wrapper import Motti4DLL


import lukefi.metsi.domain.natural_processes.grow_motti_dll as grow_motti
from lukefi.metsi.domain.natural_processes.motti_dll_wrapper import GrowthDeltas
from lukefi.metsi.data.enums.internal import DrainageCategory


from lukefi.metsi.domain.natural_processes.grow_motti_dll import (
    resolve_shared_object,
    resolve_dir_or_file,
    default_data_dir,
    find_repo_root,
)


def make_empty_ut_buffers() -> Any:
    blank_species = SimpleNamespace(
        year=-1.0,
        f_kkp=0.0, f_klv=0.0, f_vlj=0.0,
        osid_kkp=0.0, osid_klv=0.0, osid_vlj=0.0,
        N_kkp=0.0, N_klv=0.0, N_vlj=0.0,
        h_kkp=0.0, h_klv=0.0, h_vlj=0.0,
        d_kkp=0.0, d_klv=0.0, d_vlj=0.0,
        age_kkp=0.0, age_klv=0.0, age_vlj=0.0,
        age13_kkp=0.0, age13_klv=0.0, age13_vlj=0.0,
        g_kkp=0.0, g_klv=0.0, g_vlj=0.0,
        v_kkp=0.0, v_klv=0.0, v_vlj=0.0,
    )
    blank_layer = SimpleNamespace(
        ma=blank_species, ku=blank_species, ra=blank_species,
        hi=blank_species, ha=blank_species, hl=blank_species,
        tl=blank_species, mh=blank_species, ml=blank_species,
        _10=blank_species,
    )
    return [[blank_layer for _ in range(10)]]


def make_empty_sapling() -> SimpleNamespace:
    """Minimal SoA-style 'sapling' container to satisfy any code that inspects it."""
    return SimpleNamespace(
        size=0,
        stems_per_ha=np.array([], dtype=float),
        breast_height_diameter=np.array([], dtype=float),
        height=np.array([], dtype=float),
        species=np.array([], dtype=int),
        biological_age=np.array([], dtype=float),
        breast_height_age=np.array([], dtype=float),
        crown_ratio=np.array([], dtype=float),
        origin=np.array([], dtype=int),
        tree_number=np.array([], dtype=int),
        stratum=np.array([], dtype=str),
    )


def make_stand_vec(rt: SimpleNamespace) -> SimpleNamespace:
    sap = make_empty_sapling()
    return SimpleNamespace(
        year=2000,
        geo_location=(6900000.0, 3400000.0, 150.0),
        lake_effect=0.0,
        sea_effect=0.0,
        land_use_category=SimpleNamespace(value=1),
        site_type_category=SimpleNamespace(value=3),
        soil_peatland_category=SimpleNamespace(value=1),
        tax_class=1,
        tax_class_reduction=0,
        reference_trees=rt,
        sapling=sap,
        saplings=sap,
        start_time=2025,
        artificial_regeneration_year=1,
        soil_surface_preparation_year=2,
        regeneration_area_cleaning_year=3,
        stand_id=12345,
        method_of_last_cutting=5,
        fertilization_year=100,
        young_stand_tending_year=200,
        drainage_category=DrainageCategory.UNDRAINED_MINERAL_SOIL_OR_MIRE,
        drainage_year=3,
        stratum="123",
    )


def make_rt(
    stems=(100.0, 120.0),
    d=(10.0, 12.0),
    h=(12.0, 14.0),
    species=(3, 7),          # 7 -> 6 (alder collapse)
    bio_age=(30.0, 32.0),
    bh_age=(20.0, 22.0),
    crown_ratio=(0.3, 0.4),
    origin=(0, 0),
):
    """Create a simple SoA (vector) reference tree container with required fields."""
    stems = np.asarray(stems, dtype=float)
    d = np.asarray(d, dtype=float)
    h = np.asarray(h, dtype=float)
    species = np.asarray(species, dtype=int)
    bio_age = np.asarray(bio_age, dtype=float)
    bh_age = np.asarray(bh_age, dtype=float)
    crown_ratio = np.asarray(crown_ratio, dtype=float)
    origin = np.asarray(origin, dtype=int)

    n = stems.shape[0]
    tree_number = np.arange(1, n + 1, dtype=int)
    stratum = np.asarray(origin, dtype=str)
    sapling = h < 1.3

    return SimpleNamespace(
        size=n,
        stems_per_ha=stems,
        breast_height_diameter=d,
        height=h,
        species=species,
        biological_age=bio_age,
        breast_height_age=bh_age,
        crown_ratio=crown_ratio,
        origin=origin,
        tree_number=tree_number,
        sapling=sapling,
        stratum=stratum,
    )
# ---------- DLL stub ----------


class FakeDLL:
    """
    Minimal DLL stub implementing the methods used by MottiDLLPredictor.
    Kept as its own concrete type so tests can access .captured_trees_py.
    """

    def __init__(self) -> None:
        self.captured_trees_py: List[Dict[str, Any]] | None = None
        self.captured_site: Dict[str, Any] | None = None
        self.captured_strata_py: List[Dict[str, Any]] | None = None

    def new_site(self, **kwargs: Any) -> SimpleNamespace:
        self.captured_site = dict(kwargs)
        return SimpleNamespace(site="ok", year=kwargs.get("year", 0), step=kwargs.get("step", 0))

    def new_trees(self, trees_py: List[Dict[str, Any]]) -> tuple[Any, int]:
        self.captured_trees_py = list(trees_py)

        yp_trees = []
        for t in trees_py:
            yp_trees.append(
                SimpleNamespace(
                    id=float(t.get("id", 0)),
                    sid=float(t.get("sid", 0)),
                    f=float(t.get("f", 0)),
                    d13=float(t.get("d13", 0)),
                    h=float(t.get("h", 0)),
                    spe=float(t.get("spe", 0)),
                    age=float(t.get("age", 0)),
                    age13=float(t.get("age13", 0)),
                    cr=float(t.get("cr", 0)),
                    snt=float(t.get("snt", 0)),
                    ba=float(t.get("ba", 0)),
                    vol=float(t.get("vol", 0)),
                )
            )

        yp = [yp_trees]
        return yp, len(trees_py)

    def new_strata(self, strata_py: List[Dict[str, Any]]) -> SimpleNamespace:
        self.captured_strata_py = list(strata_py)
        return SimpleNamespace(strata="ok")

    def alloc_state_buffers(self, ctrl: Any = None) -> Any:
        return SimpleNamespace(
            buffers="ok",
            ctrl=ctrl,
            saplings=make_empty_ut_buffers(),
        )

    def initialize_with_state(
        self,
        yo: Any,
        yy: Any,
        yp: Any,
        numtrees: int,
        buffers: Any,
    ) -> int:
        return int(numtrees)

    def grow_with_state(self, *_args: Any, **_kwargs: Any) -> GrowthDeltas:
        if not self.captured_trees_py:
            return GrowthDeltas(
                tree_ids=[],
                trees_id=[],
                trees_ih=[],
                trees_if=[],
                trees_age=[],
                trees_age13=[],
            )

        n = len(self.captured_trees_py)
        ids = list(range(1, n + 1))
        zeros = [0.0] * n
        return GrowthDeltas(
            tree_ids=ids,
            trees_id=zeros,
            trees_ih=zeros,
            trees_if=zeros,
            trees_age=zeros,
            trees_age13=zeros,
        )

# ---------- Tests ----------


class TestMottiPathResolversAndWrapperUtils(unittest.TestCase):
    def setUp(self):
        self._old_env = dict(os.environ)

    def tearDown(self):
        # restore environment as it was
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_resolve_shared_object_exact_file(self):
        with tempfile.TemporaryDirectory() as td:
            lib = Path(td) / "libmottisc.so"
            lib.write_text("")  # create empty placeholder
            out = resolve_shared_object(lib)
            self.assertTrue(out.is_file())
            self.assertEqual(out.resolve(), lib.resolve())

    def test_resolve_shared_object_discovers_in_dir(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            target = base / "libmottisc.so"
            target.write_text("")
            out = resolve_shared_object(base)
            self.assertEqual(out.resolve(), target.resolve())

    def test_resolve_shared_object_no_match_returns_dir(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            out = resolve_shared_object(base)
            # No candidate found → function returns the directory for downstream to error clearly
            self.assertTrue(out.is_dir())
            self.assertEqual(out.resolve(), base.resolve())

    def test_resolve_shared_object_none_raises(self):
        # type: ignore to call with None on purpose (function raises by design)
        with self.assertRaises(ValueError):
            resolve_shared_object(None)  # type: ignore[arg-type]

    def test_resolve_dir_or_file_none_uses_default_env_override(self):
        with tempfile.TemporaryDirectory() as td:
            override = Path(td) / "data" / "motti"
            override.mkdir(parents=True, exist_ok=True)
            os.environ["MOTTI_DATA_DIR"] = str(override)
            out = resolve_dir_or_file(None)
            self.assertEqual(out, override.resolve())

    def test_resolve_dir_or_file_relative_and_tilde(self):
        with tempfile.TemporaryDirectory() as td:
            # Relative path becomes absolute from CWD
            cwd = Path.cwd()
            try:
                os.chdir(td)
                rel = Path("some/dir")
                rel.mkdir(parents=True, exist_ok=True)
                out_rel = resolve_dir_or_file("some/dir")
                self.assertEqual(out_rel, (Path(td) / "some" / "dir").resolve())
            finally:
                os.chdir(cwd)

            # Tilde + env expansion (cross-platform)
            home_dir = Path(td) / "homeA"
            (home_dir / "x").mkdir(parents=True, exist_ok=True)
            # Make expanduser('~') point to our temp home on all OSes:
            os.environ["HOME"] = str(home_dir)
            os.environ["USERPROFILE"] = str(home_dir)  # Windows prefers this
            os.environ.pop("HOMEDRIVE", None)          # Avoid legacy precedence
            os.environ.pop("HOMEPATH", None)
            expected = (Path(os.path.expanduser("~")) / "x").resolve()
            out_tilde = resolve_dir_or_file("~/x")
            self.assertEqual(out_tilde.resolve(), expected)

    def test_default_data_dir_prefers_repo_root_or_env(self):
        # 1) With MOTTI_DATA_DIR set
        with tempfile.TemporaryDirectory() as td:
            override = Path(td) / "over" / "ride"
            override.mkdir(parents=True, exist_ok=True)
            os.environ["MOTTI_DATA_DIR"] = str(override)
            self.assertEqual(default_data_dir(), override.resolve())
            os.environ.pop("MOTTI_DATA_DIR", None)

        # 2) No env: it should pick {repo_root}/data/motti; validate root detection
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "data" / "motti").mkdir(parents=True, exist_ok=True)
            # Create a nested child and simulate running from there
            child = root / "nested" / "deeper"
            child.mkdir(parents=True, exist_ok=True)
            cwd = Path.cwd()
            try:
                os.chdir(child)
                # Because repo root is discovered, default dir should be root/data/motti
                out = default_data_dir()
                self.assertEqual(out, (root / "data" / "motti").resolve())
            finally:
                os.chdir(cwd)

    def test_find_repo_root_markers(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            # Any of these should count; try pyproject first
            (base / "pyproject.toml").write_text("[tool.poetry]\nname='x'\n")
            self.assertEqual(find_repo_root(base / "a" / "b" / "c"), base.resolve())

            # Try .git marker
            (base / "pyproject.toml").unlink()
            (base / ".git").mkdir()
            self.assertEqual(find_repo_root(base / "x"), base.resolve())

            # Try data/motti marker
            for p in (base / ".git",):
                if p.exists() and p.is_dir():
                    for _child in p.iterdir():
                        pass
                # clean .git directory
            # Safer: use a fresh temp directory for clarity
        with tempfile.TemporaryDirectory() as td2:
            base2 = Path(td2)
            (base2 / "data" / "motti").mkdir(parents=True, exist_ok=True)
            self.assertEqual(find_repo_root(base2 / "n1" / "n2"), base2.resolve())

    def test_wrapper_maybe_chdir_changes_cwd_temporarily(self):
        start = Path.cwd().resolve()
        with tempfile.TemporaryDirectory() as td:
            target = Path(td).resolve()
            with Motti4DLL.maybe_chdir(target):
                self.assertEqual(Path.cwd().resolve(), target)
            # back to original
            self.assertEqual(Path.cwd().resolve(), start)


class TestGrowMottiDLLVec(unittest.TestCase):
    def test_species_mapping_and_euref(self) -> None:
        # species mapping: alder collapse (7 -> 6); others pass-through or bucketed
        self.assertEqual(grow_motti.species_to_motti(7), 6)
        self.assertEqual(grow_motti.species_to_motti(3), 3)

        # auto_euref_km conversion logic
        y_km, x_km = grow_motti.auto_euref_km(6900000.0, 3400000.0)
        self.assertEqual((y_km, x_km), (6900.0, 3400.0))
        y_10km, x_10km = grow_motti.auto_euref_km(6900.0, 3400.0)
        self.assertEqual((y_10km, x_10km), (6.9, 3.4))
        with self.assertRaises(ValueError):
            grow_motti.auto_euref_km(62.0, 25.0)  # looks like lat/lon -> should raise

    def test_predictor_builds_tree_payload_and_species_mapping(self) -> None:
        rt = make_rt(species=(3, 7))  # 7 -> 6
        stand = make_stand_vec(rt)

        dll_stub = FakeDLL()
        # grow_motti.MottiDLLPredictorVec expects a Motti4DLL, but our stub is duck-typed.
        pred = grow_motti.MottiDLLPredictor(stand, dll=dll_stub)  # type: ignore[arg-type]

        # Run evolve once to populate the payload
        _ = pred.evolve(step=5, sim_year=stand.year)

        trees_py = dll_stub.captured_trees_py
        self.assertIsNotNone(trees_py, "DLL tree payload was not captured by stub")
        assert trees_py is not None  # for type checkers
        self.assertEqual(trees_py[0]["id"], 1)
        self.assertEqual(trees_py[1]["id"], 2)
        self.assertEqual(trees_py[0]["spe"], 3)
        self.assertEqual(trees_py[1]["spe"], 6)  # alder collapsed

    def test_vector_grow_applies_deltas_and_handles_deaths(self) -> None:
        # Two trees; DLL returns growth only for tree 1; tree 2 "dies" (missing -> stems=0)
        rt = make_rt(
            stems=(100.0, 80.0),
            d=(10.0, 12.0),
            h=(12.0, 14.0),
            species=(2, 3),
        )
        stand = make_stand_vec(rt)

        class GrowingDLL(FakeDLL):
            def grow_with_state(self, *args: Any, **kwargs: Any) -> GrowthDeltas:  # noqa: D401
                # Only tree id=1 grows / survives
                return GrowthDeltas(
                    tree_ids=[1],
                    trees_id=[+0.7],    # Δd
                    trees_ih=[+1.2],    # Δh
                    trees_if=[-5.0],    # Δf
                    trees_age=[20],
                    trees_age13=[10],
                )

        dll_stub = GrowingDLL()
        pred = grow_motti.MottiDLLPredictor(stand, dll=dll_stub)  # type: ignore[arg-type]

        out_stand, _ = grow_motti.grow_motti_dll_fn(
            stand,  # type: ignore[arg-type]
            predictor=pred,
            step=5,
        )

        # Make linters happy: ensure we got a vector trees container back
        self.assertIsNotNone(out_stand.reference_trees)
        rt_out = out_stand.reference_trees
        assert rt_out is not None

        # tree 1 updated by deltas
        self.assertAlmostEqual(rt_out.breast_height_diameter[0], 10.0 + 0.7, places=6)
        self.assertAlmostEqual(rt_out.height[0], 12.0 + 1.2, places=6)
        self.assertAlmostEqual(rt_out.stems_per_ha[0], 100.0 - 5.0, places=6)

        # tree 2 missing from DLL result → stems set to 0 (d, h unchanged)
        self.assertAlmostEqual(rt_out.breast_height_diameter[1], 12.0, places=6)
        self.assertAlmostEqual(rt_out.height[1], 14.0, places=6)
        self.assertEqual(float(rt_out.stems_per_ha[1]), 0.0)
