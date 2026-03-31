import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from lukefi.metsi.data.model import ForestStand, MottiState
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.data.enums.internal import (
    DrainageCategory,
    LandUseCategory,
    SiteType,
    SoilPeatlandCategory,
    TreeSpecies,
)
from lukefi.metsi.domain.natural_processes.grow_motti_dll import (
    apply_motti_yp_reduction_from_removed_reference_trees,
    sync_ut_to_reference_trees,
    sync_yp_to_reference_trees,
)


def _blank_species() -> SimpleNamespace:
    return SimpleNamespace(
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


def _empty_ut_buffers() -> Any:
    layer = SimpleNamespace(
        ma=_blank_species(), ku=_blank_species(), ra=_blank_species(),
        hi=_blank_species(), ha=_blank_species(), hl=_blank_species(),
        tl=_blank_species(), mh=_blank_species(), ml=_blank_species(),
        _10=_blank_species(),
    )
    return [[layer for _ in range(10)]]


def _make_stand() -> ForestStand:
    stand = ForestStand(
        identifier="stand-1",
        time=2025,
        start_time=2025,
        stand_id=123,
        geo_location=(6900000.0, 3400000.0, 150.0, "EPSG:3067"),
        land_use_category=LandUseCategory(1),
        site_type_category=SiteType(3),
        soil_peatland_category=SoilPeatlandCategory(1),
        drainage_category=DrainageCategory.UNDRAINED_MINERAL_SOIL_OR_MIRE,
        tax_class=1,
        tax_class_reduction=0,
    )
    stand.reference_trees = ReferenceTrees()
    stand.reference_trees.create([
        {
            "identifier": "stand-1-1-tree",
            "tree_number": 1,
            "species": int(TreeSpecies.PINE),
            "stems_per_ha": 120.0,
            "breast_height_diameter": 22.0,
            "height": 18.0,
            "biological_age": 40.0,
            "breast_height_age": 25.0,
            "origin": 0,
            "sapling": False,
            "tree_category": "1",
            "management_category": 1,
            "stratum": "101",
        },
        {
            "identifier": "stand-1-2-tree",
            "tree_number": 2,
            "species": int(TreeSpecies.SPRUCE),
            "stems_per_ha": 80.0,
            "breast_height_diameter": 18.0,
            "height": 15.0,
            "biological_age": 35.0,
            "breast_height_age": 20.0,
            "origin": 0,
            "sapling": False,
            "tree_category": "1",
            "management_category": 1,
            "stratum": "102",
        },
    ])

    stand.tree_strata = TreeStrata(size=1)
    stand.tree_strata.storey[0] = 2
    stand.tree_strata.identifier[0] = "layer-1"
    return stand


class SpyDLL:
    def __init__(self) -> None:
        self.update_after_import_calls = 0
        self.grow_zero_calls = 0
        self.regenerate_calls = 0

    def update_after_import(self, yy: Any, yp: Any, numtrees: int, buffers: Any) -> int:
        self.update_after_import_calls += 1
        return int(numtrees)

    def grow_with_state(self, yy: Any, yp: Any, numtrees: int, buffers: Any, *, step: int = 0):
        if step == 0:
            self.grow_zero_calls += 1
        return SimpleNamespace(
            tree_ids=[int(yp[0][i].id) for i in range(int(numtrees))],
            trees_id=[0.0] * int(numtrees),
            trees_ih=[0.0] * int(numtrees),
            trees_if=[0.0] * int(numtrees),
            trees_age=[float(yp[0][i].age) for i in range(int(numtrees))],
            trees_age13=[float(yp[0][i].age13) for i in range(int(numtrees))],
        )

    def regenerate_with_state(
        self,
        yy: Any,
        yp: Any,
        numtrees: int,
        buffers: Any,
        *,
        method: list[float],
        step: int = 0,
    ) -> int:
        self.regenerate_calls += 1
        s = buffers.saplings[0][0].ma
        s.year = 2025.0
        s.f_kkp = 1500.0
        s.osid_kkp = 201.0
        s.N_kkp = 2.0
        s.h_kkp = 0.7
        s.d_kkp = 0.0
        s.age_kkp = 3.0
        s.age13_kkp = 0.0
        s.g_kkp = 0.0
        s.v_kkp = 0.0
        return int(numtrees)


class TestMottiSyncNewBehaviour(unittest.TestCase):
    def _attach_state(self, stand: ForestStand) -> SpyDLL:
        dll = SpyDLL()
        yp = [[
            SimpleNamespace(
                id=1.0, sid=101.0, f=120.0, d13=22.0, h=18.0,
                spe=float(int(TreeSpecies.PINE)), age=40.0, age13=25.0,
                snt=1.0, ba=0.5, vol=1.2,
            ),
            SimpleNamespace(
                id=2.0, sid=102.0, f=80.0, d13=18.0, h=15.0,
                spe=float(int(TreeSpecies.SPRUCE)), age=35.0, age13=20.0,
                snt=1.0, ba=0.3, vol=0.8,
            ),
        ]]
        stand.motti_state = MottiState(
            dll=dll,
            yy=SimpleNamespace(year=2025.0, step=0.0),
            yp=yp,
            ntrees=2,
            buffers=SimpleNamespace(saplings=_empty_ut_buffers()),
            signature=(1, 2),
        )
        return dll

    def test_sync_ut_to_reference_trees_creates_new_sapling_reference_tree(self) -> None:
        stand = _make_stand()
        self._attach_state(stand)

        ut = stand.motti_state.buffers.saplings[0][0].ma
        ut.year = 2025.0
        ut.f_kkp = 1500.0
        ut.osid_kkp = 201.0
        ut.N_kkp = 2.0
        ut.h_kkp = 0.7
        ut.d_kkp = 0.0
        ut.age_kkp = 3.0
        ut.age13_kkp = 0.0
        ut.g_kkp = 0.0
        ut.v_kkp = 0.0

        sync_ut_to_reference_trees(stand)

        self.assertEqual(stand.reference_trees.size, 3)
        idx = 2
        self.assertEqual(stand.reference_trees.stratum[idx], "201")
        self.assertEqual(int(stand.reference_trees.species[idx]), int(TreeSpecies.PINE))
        self.assertTrue(bool(stand.reference_trees.sapling[idx]))
        self.assertAlmostEqual(float(stand.reference_trees.stems_per_ha[idx]), 1500.0)
        self.assertAlmostEqual(float(stand.reference_trees.height[idx]), 0.7)

    def test_sync_yp_to_reference_trees_updates_existing_big_tree_by_sid(self) -> None:
        stand = _make_stand()
        self._attach_state(stand)

        t0 = stand.motti_state.yp[0][0]
        t0.f = 95.0
        t0.d13 = 23.5
        t0.h = 18.6
        t0.age = 41.0
        t0.age13 = 26.0
        t0.ba = 0.61
        t0.vol = 1.45

        sync_yp_to_reference_trees(stand)

        self.assertAlmostEqual(float(stand.reference_trees.stems_per_ha[0]), 95.0)
        self.assertAlmostEqual(float(stand.reference_trees.breast_height_diameter[0]), 23.5)
        self.assertAlmostEqual(float(stand.reference_trees.height[0]), 18.6)
        self.assertFalse(bool(stand.reference_trees.sapling[0]))

    def test_cutting_side_reduction_is_written_to_yp_and_synced_back(self) -> None:
        stand = _make_stand()
        dll = self._attach_state(stand)

        removed_f = np.array([20.0, 0.0], dtype=float)
        changed = apply_motti_yp_reduction_from_removed_reference_trees(stand, removed_f, refresh=False)

        self.assertTrue(changed)
        self.assertAlmostEqual(float(stand.motti_state.yp[0][0].f), 100.0)
        self.assertAlmostEqual(float(stand.motti_state.yp[0][1].f), 80.0)

        sync_yp_to_reference_trees(stand)
        self.assertAlmostEqual(float(stand.reference_trees.stems_per_ha[0]), 100.0)
        self.assertEqual(dll.grow_zero_calls, 0)

    def test_regeneration_path_can_be_verified_with_regenerate_stub_and_ut_sync(self) -> None:
        stand = _make_stand()
        dll = self._attach_state(stand)

        dll.regenerate_with_state(
            stand.motti_state.yy,
            stand.motti_state.yp,
            stand.motti_state.ntrees,
            stand.motti_state.buffers,
            method=[3.0, 100.0, 1.0, 1500.0, 0.0, 0.0, 0.0],
            step=0,
        )
        sync_ut_to_reference_trees(stand)

        self.assertEqual(dll.regenerate_calls, 1)
        self.assertEqual(stand.reference_trees.size, 3)
        self.assertEqual(stand.reference_trees.stratum[2], "201")
        self.assertEqual(int(stand.reference_trees.species[2]), int(TreeSpecies.PINE))
        self.assertTrue(bool(stand.reference_trees.sapling[2]))


class TestKnownGapInCurrentImplementation(unittest.TestCase):
    @unittest.expectedFailure
    def test_refresh_after_python_side_yp_edit_should_call_update_after_import_before_zero_step_growth(self) -> None:
        stand = _make_stand()
        dll = SpyDLL()
        stand.motti_state = MottiState(
            dll=dll,
            yy=SimpleNamespace(year=2025.0, step=0.0),
            yp=[[SimpleNamespace(id=1.0, sid=101.0, f=100.0, d13=22.0, h=18.0,
                                 spe=float(int(TreeSpecies.PINE)), age=40.0, age13=25.0,
                                 snt=1.0, ba=0.5, vol=1.2)]],
            ntrees=1,
            buffers=SimpleNamespace(saplings=_empty_ut_buffers()),
            signature=(1,),
        )

        removed_f = np.array([10.0, 0.0], dtype=float)
        apply_motti_yp_reduction_from_removed_reference_trees(stand, removed_f, refresh=True)

        # Current code performs zero-step grow, but skips dll.update_after_import().
        self.assertEqual(dll.update_after_import_calls, 1)
        self.assertEqual(dll.grow_zero_calls, 1)


if __name__ == "__main__":
    unittest.main()
