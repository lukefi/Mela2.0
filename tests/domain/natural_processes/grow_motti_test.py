import os
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch
import numpy as np
from lukefi.metsi.data.enums.internal import (
    CRS,
    DrainedPeatlandForestType,
    LandUseCategory,
    SiteType,
    SoilPeatlandCategory,
    TreeSpecies,
    DrainageCategory)
from lukefi.metsi.data.conversion import internal2motti
from lukefi.metsi.data.motti.motti_types import MottiState
from lukefi.metsi.domain.natural_processes import grow_motti, motti_initialization
from lukefi.metsi.forestry.naturalprocess.motti_dll_wrapper import (
    Motti4DLL,
    _default_data_dir,
    _maybe_chdir,
    _resolve_dir_or_file,
    _resolve_shared_object)
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata


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


def _make_stand_vec(rt: ReferenceTrees) -> SimpleNamespace:
    sap = make_empty_sapling()
    return SimpleNamespace(
        identifier="stand-12345",
        motti_state=None,
        time=2000,
        year=2000,
        relative_year=2000,
        geo_location=(6900000.0, 3400000.0, 150.0, CRS.EPSG_3067),
        lake_effect=0.0,
        sea_effect=0.0,
        land_use_category=LandUseCategory.FOREST,
        site_type_category=SiteType.DAMP_SITE,
        soil_peatland_category=SoilPeatlandCategory.MINERAL_SOIL,
        tax_class=1,
        tax_class_reduction=0,
        reference_trees=rt,
        tree_strata=TreeStrata(),
        sapling=sap,
        saplings=sap,
        start_time=2025,
        artificial_regeneration_year=1,
        soil_surface_preparation_year=2,
        regeneration_area_cleaning_year=3,
        stand_id=12345,
        cutting_year=1999,
        method_of_last_cutting=5,
        fertilization_year=100,
        young_stand_tending_year=200,
        drainage_category=DrainageCategory.UNDRAINED_MINERAL_SOIL_OR_MIRE,
        drained_peatland_type=None,
        drainage_year=3,
        stratum="123",
        start_year=2025,
    )


def _make_rt(
    stems=(100.0, 120.0),
    d=(10.0, 12.0),
    h=(12.0, 14.0),
    species=(3, 7),
    bio_age=(30.0, 32.0),
    bh_age=(20.0, 22.0),
    crown_ratio=(0.3, 0.4),
    origin=(0, 0),
    storey=(0, 0),
) -> ReferenceTrees:
    """Build a real ReferenceTrees vector container for Motti sync tests.

    The Motti reconciliation path now calls ReferenceTrees.create/update/delete
    and also allocates new identifiers from stand.identifier. Returning the
    production vector container keeps this fixture aligned with that API.
    """
    stems_arr = np.asarray(stems, dtype=float)
    d_arr = np.asarray(d, dtype=float)
    h_arr = np.asarray(h, dtype=float)
    species_arr = np.asarray(species, dtype=int)
    bio_age_arr = np.asarray(bio_age, dtype=float)
    bh_age_arr = np.asarray(bh_age, dtype=float)
    crown_ratio_arr = np.asarray(crown_ratio, dtype=float)
    origin_arr = np.asarray(origin, dtype=int)
    storey_arr = np.asarray(storey, dtype=int)

    n = stems_arr.shape[0]
    rt = ReferenceTrees()
    rt.create([
        {
            "identifier": f"stand-12345-{i + 1}-tree",
            "tree_number": i + 1,
            "stems_per_ha": float(stems_arr[i]),
            "breast_height_diameter": float(d_arr[i]),
            "height": float(h_arr[i]),
            "species": int(species_arr[i]),
            "biological_age": float(bio_age_arr[i]),
            "breast_height_age": float(bh_age_arr[i]),
            "crown_ratio": float(crown_ratio_arr[i]),
            "origin": int(origin_arr[i]),
            "stratum": int(origin_arr[i]),
            "storey": int(storey_arr[i]),
            "sapling": bool(h_arr[i] < 1.3),
            "tree_category": "1",
            "management_category": 1,
        }
        for i in range(n)
    ])
    return rt

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

    def alloc_state_buffers(self) -> Any:
        return SimpleNamespace(
            buffers="ok",
            ctrl=SimpleNamespace(death_tree=1),
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

    def grow_with_state(self, *_args: Any, **_kwargs: Any):
        return

    def grow(self, *_args: Any, **_kwargs: Any):
        return self.grow_with_state(*_args, **_kwargs)


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
            out = _resolve_shared_object(lib)
            self.assertTrue(out.is_file())
            self.assertEqual(out.resolve(), lib.resolve())

    def test_resolve_shared_object_discovers_in_dir(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            target = base / "libmottisc.so"
            target.write_text("")
            out = _resolve_shared_object(base)
            self.assertEqual(out.resolve(), target.resolve())

    def test_resolve_shared_object_no_match_returns_dir(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            out = _resolve_shared_object(base)
            # No candidate found → function returns the directory for downstream to error clearly
            self.assertTrue(out.is_dir())
            self.assertEqual(out.resolve(), base.resolve())

    def test_resolve_dir_or_file_none_uses_default_env_override(self):
        with tempfile.TemporaryDirectory() as td:
            override = Path(td) / "data" / "motti"
            override.mkdir(parents=True, exist_ok=True)
            os.environ["MOTTI_DATA_DIR"] = str(override)
            out = _resolve_dir_or_file(None)
            self.assertEqual(out, override.resolve())

    def test_resolve_dir_or_file_relative_and_tilde(self):
        with tempfile.TemporaryDirectory() as td:
            # Relative path becomes absolute from CWD
            cwd = Path.cwd()
            try:
                os.chdir(td)
                rel = Path("some/dir")
                rel.mkdir(parents=True, exist_ok=True)
                out_rel = _resolve_dir_or_file("some/dir")
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
            out_tilde = _resolve_dir_or_file("~/x")
            self.assertEqual(out_tilde.resolve(), expected)

    def test_default_data_dir_prefers_repo_root_or_env(self):
        # 1) With MOTTI_DATA_DIR set
        with tempfile.TemporaryDirectory() as td:
            override = Path(td) / "over" / "ride"
            override.mkdir(parents=True, exist_ok=True)
            os.environ["MOTTI_DATA_DIR"] = str(override)
            self.assertEqual(_default_data_dir(), override.resolve())
            os.environ.pop("MOTTI_DATA_DIR", None)

        # 2) No env: it should pick {repo_root}/data/motti
        # Because repo root is discovered, default dir should be root/data/motti
        out = _default_data_dir()
        self.assertEqual(out, (Path.cwd() / "data" / "motti").resolve())

    def test_wrapper_maybe_chdir_changes_cwd_temporarily(self):
        start = Path.cwd().resolve()
        with tempfile.TemporaryDirectory() as td:
            target = Path(td).resolve()
            with _maybe_chdir(target):
                self.assertEqual(Path.cwd().resolve(), target)
            # back to original
            self.assertEqual(Path.cwd().resolve(), start)


class TestGrowMottiDLLVec(unittest.TestCase):

    def test_species_mapping_and_euref(self) -> None:
        # NOTE: Siis eikö tämän kannattas olla jossain tests\...\enum hakemistossa?
        # species mapping: alder collapse (7 -> 6); others pass-through or bucketed
        self.assertEqual(internal2motti.convert_species(TreeSpecies(7)), 6)
        self.assertEqual(internal2motti.convert_species(TreeSpecies(3)), 3)

        # auto_euref_km conversion logic
        _geo_location = (6900000.0, 3400000.0, None, CRS.EPSG_3067)
        y_km, x_km = motti_initialization._auto_euref_km(_geo_location) # pylint: disable=protected-access
        self.assertEqual((y_km, x_km), (6900.0, 3400.0))
        with self.assertRaises(ValueError):
            _geo_location = (None, None, None, CRS.EPSG_2393)
            motti_initialization._auto_euref_km(_geo_location)  # pylint: disable=protected-access

    def test_predictor_builds_tree_payload_and_species_mapping(self) -> None:
        rt = _make_rt(species=(3, 7))  # 7 -> 6
        stand = _make_stand_vec(rt)
        fake_dll = FakeDLL()

        # grow_motti.MottiDLLPredictorVec expects a Motti4DLL, but our stub is duck-typed.
        with (patch.object(Motti4DLL, "new_site", fake_dll.new_site),
              patch.object(Motti4DLL, "new_trees", fake_dll.new_trees),
              patch.object(Motti4DLL, "new_strata", fake_dll.new_strata),
              patch.object(Motti4DLL, "alloc_state_buffers", fake_dll.alloc_state_buffers),
              patch.object(Motti4DLL, "initialize_with_state", fake_dll.initialize_with_state),
              patch.object(Motti4DLL, "grow_with_state", fake_dll.grow_with_state)):
            stand.motti_state = motti_initialization._init_motti_state(stand) # pylint: disable=protected-access
            grow_motti.grow_motti_fn(stand, step=5)

        trees_py = fake_dll.captured_trees_py
        self.assertIsNotNone(trees_py, "DLL tree payload was not captured by stub")
        assert trees_py is not None  # for type checkers
        self.assertEqual(trees_py[0]["id"], 1)
        self.assertEqual(trees_py[1]["id"], 2)
        self.assertEqual(trees_py[0]["spe"], 3)
        self.assertEqual(trees_py[1]["spe"], 6)  # alder collapsed

    def test_vector_grow_applies_deltas_and_handles_deaths(self) -> None:
        # Two trees; DLL returns growth only for tree 1; tree 2 "dies" (missing -> stems=0)
        rt = _make_rt(
            stems=(100.0, 80.0),
            d=(10.0, 12.0),
            h=(12.0, 14.0),
            species=(2, 3),
            origin=(2, 2),
        )
        stand = _make_stand_vec(rt)

        class GrowingDLL(FakeDLL):
            def grow_with_state(self, state: MottiState, **kwargs: Any):  # noqa: D401
                _ = kwargs
                yp = state.yp
                surviving = yp[0][0]
                surviving.d13 += 0.7
                surviving.h += 1.2
                surviving.f -= 5.0
                surviving.age = 20.0
                surviving.age13 = 10.0
                state.ntrees = 1

        dll_stub = GrowingDLL()

        with (patch.object(Motti4DLL, "new_site", dll_stub.new_site),
              patch.object(Motti4DLL, "new_trees", dll_stub.new_trees),
              patch.object(Motti4DLL, "new_strata", dll_stub.new_strata),
              patch.object(Motti4DLL, "alloc_state_buffers", dll_stub.alloc_state_buffers),
              patch.object(Motti4DLL, "initialize_with_state", dll_stub.initialize_with_state),
              patch.object(Motti4DLL, "grow_with_state", dll_stub.grow_with_state)):
            stand.motti_state = motti_initialization._init_motti_state(stand)  # pylint: disable=protected-access
            out_stand, _ = grow_motti.grow_motti_fn(
                stand,
                step=5,
            )

        rt_out = out_stand.reference_trees
        assert rt_out is not None

        # tree 1 updated through the mutated Motti yp buffer
        self.assertEqual(rt_out.size, 1)
        self.assertAlmostEqual(rt_out.breast_height_diameter[0], 10.0 + 0.7, places=6)
        self.assertAlmostEqual(rt_out.height[0], 12.0 + 1.2, places=6)
        self.assertAlmostEqual(rt_out.stems_per_ha[0], 100.0 - 5.0, places=6)

        # tree 2 missing from DLL result → pruned by reconcile_reference_trees_from_motti
