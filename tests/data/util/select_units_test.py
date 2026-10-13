import unittest
import numpy as np
import pandas as pd
from lukefi.metsi.data.enums.internal import (
    CrownClass,
    CuttingMethod,
    DamageType,
    DrainageCategory,
    LandUseCategory,
    Origin,
    OwnerCategory,
    SiteType,
    SoilPeatlandCategory,
    Storey,
    TreeCategory,
    TreeManagementCategory,
    TreeType)
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.core.select_units import (
    Mode,
    ProfileXMode,
    SelectionSet,
    SelectionTarget,
    TargetType,
    select_units)
from lukefi.metsi.data.vector_model import ReferenceTrees


class TestSelectUnits(unittest.TestCase):

    stand: ForestStand
    trees: ReferenceTrees

    def setUp(self) -> None:
        self.stand = ForestStand()
        self.stand.stand_id = 3
        self.stand.year = 2022
        self.stand.area = 2.81114
        self.stand.area_weight = 2.81114
        self.stand.geo_location = (6643.59, 274.542, 11.080000, None)
        self.stand.owner_category = OwnerCategory(0)
        self.stand.land_use_category = LandUseCategory(1)
        self.stand.soil_peatland_category = SoilPeatlandCategory(1)
        self.stand.site_type_category = SiteType(1)
        self.stand.tax_class_reduction = 0
        self.stand.drainage_category = DrainageCategory(2)
        self.stand.drainage_year = 0
        self.stand.fertilization_year = 0
        self.stand.soil_surface_preparation_year = 2023
        self.stand.regeneration_area_cleaning_year = 0
        self.stand.young_stand_tending_year = 0
        self.stand.cutting_year = 0
        self.stand.forest_management_category = 1
        self.stand.method_of_last_cutting = CuttingMethod.NO_CUTTING
        self.stand.municipality_id = 78
        self.stand.forestry_centre_id = 1

        pd_trees = pd.read_csv(
            "tests/resources/trees_01.csv",
            delimiter=" ",
            header=0)

        self.trees = ReferenceTrees()
        self.trees.size = len(pd_trees)
        self.trees.identifier = np.repeat("", self.trees.size)
        self.trees.tree_number = np.ascontiguousarray(pd_trees.treenr, dtype=np.int32)
        self.trees.species = np.ascontiguousarray(pd_trees.spe, dtype=np.int32)
        self.trees.breast_height_diameter = np.ascontiguousarray(pd_trees.d, dtype=np.float64)
        self.trees.height = np.ascontiguousarray(pd_trees.h, dtype=np.float64)
        self.trees.measured_height = np.repeat(0.0, self.trees.size)
        self.trees.breast_height_age = np.ascontiguousarray(pd_trees.age13, dtype=np.float64)
        self.trees.biological_age = np.ascontiguousarray(pd_trees.ageb, dtype=np.float64)
        self.trees.stems_per_ha = np.ascontiguousarray(pd_trees.f, dtype=np.float64)
        self.trees.origin = np.repeat(Origin.NATURAL, self.trees.size)
        self.trees.management_category = np.ascontiguousarray(pd_trees.manag_cat, np.int32)
        self.trees.tree_category = np.repeat(TreeCategory.UNSET, self.trees.size)
        self.trees.storey = np.repeat(Storey.INDETERMINATE, self.trees.size)
        self.trees.sapling = np.repeat(False, self.trees.size)
        self.trees.tree_type = np.repeat(TreeType.UNSET, self.trees.size)
        self.trees.damage_type = np.repeat(DamageType.UNSET, self.trees.size)
        self.trees.crown_class = np.repeat(CrownClass.CROWNLESS, self.trees.size)
        self.stand.reference_trees = self.trees

    def test_odds_units(self):
        set1 = SelectionSet[ForestStand, ReferenceTrees](
            lambda _, trees: trees.management_category <= TreeManagementCategory.NO_RESTRICTION,
            "breast_height_diameter",
            "stems_per_ha",
            TargetType.RELATIVE,
            0.5,
            np.array([0.0, 0.5, 1.0]),
            np.array([0.01, 0.5, 0.99]),
            ProfileXMode.RELATIVE
        )
        set1.sfunction = lambda _, trees: trees.management_category <= TreeManagementCategory.NO_RESTRICTION
        set1.order_var = "breast_height_diameter"
        set1.target_var = "stems_per_ha"
        set1.target_type = TargetType.RELATIVE
        set1.target_amount = 0.5
        set1.profile_x = np.array([0.0, 0.5, 1.0])
        set1.profile_y = np.array([0.01, 0.5, 0.99])
        set1.profile_xmode = ProfileXMode.RELATIVE

        sets = [set1]

        target = SelectionTarget(TargetType.ABSOLUTE_REMAIN, "stems_per_ha", 100)

        selected = select_units(self.stand, self.trees, target, sets, "stems_per_ha")
        expected = np.array([1.28591462e-03, 2.63355633e-02, 4.43962202e-02, 6.42423503e-02,
                             6.20824116e-02, 8.13241804e-02, 1.02165851e-01, 1.25481309e-01,
                             1.52254037e-01, 1.83593181e-01, 2.22116477e-01, 2.68280616e-01,
                             3.28375962e-01, 4.07667649e-01, 5.19812331e-01, 6.90396404e-01,
                             6.17950003e-01, 9.71501929e-01, 1.56268073e+00, 3.49990638e+00])
        self.assertTrue(np.all(np.abs(selected - expected) < 0.00000001))

    def test_odds_profile(self):
        set1 = SelectionSet[ForestStand, ReferenceTrees](
            lambda _, trees: trees.management_category <= TreeManagementCategory.NO_RESTRICTION,
            "breast_height_diameter",
            "stems_per_ha",
            TargetType.RELATIVE,
            0.5,
            np.array([0.0, 0.5, 1.0]),
            np.array([0.01, 0.5, 0.99]),
            ProfileXMode.RELATIVE
        )

        sets = [set1]

        target = SelectionTarget(TargetType.ABSOLUTE_REMAIN, "stems_per_ha", 100)

        selected = select_units(self.stand, self.trees, target, sets, "stems_per_ha", mode=Mode.ODDS_PROFILE)
        expected = np.array([2.17248250e-04, 6.92270568e-03, 1.02994674e-02, 1.33560941e-02,
                             1.28738888e-02, 1.51180342e-02, 5.13977933e-02, 1.84254753e-01,
                             3.06567510e-01, 4.20444904e-01, 5.30104617e-01, 6.31328967e-01,
                             7.30444477e-01, 8.25342305e-01, 9.18131293e-01, 1.00881144e+00,
                             8.50025789e-01, 1.09527391e+00, 1.18173637e+00, 1.26609000e+00])
        self.assertTrue(np.all(np.abs(selected - expected) < 0.00000001))

    def test_scale(self):
        set1 = SelectionSet[ForestStand, ReferenceTrees](
            lambda _, trees: trees.management_category <= TreeManagementCategory.NO_RESTRICTION,
            "breast_height_diameter",
            "stems_per_ha",
            TargetType.RELATIVE,
            0.5,
            np.array([0.0, 0.5, 1.0]),
            np.array([0.01, 0.5, 0.99]),
            ProfileXMode.RELATIVE
        )

        sets = [set1]

        target = SelectionTarget(TargetType.ABSOLUTE_REMAIN, "stems_per_ha", 100)

        selected = select_units(self.stand, self.trees, target, sets, "stems_per_ha", mode=Mode.SCALE)
        expected = np.array([
            0.0102433, 0.1676316, 0.2474976, 0.3199432, 0.3083750, 0.3614502, 0.4073618,
            0.4486822, 0.4867233, 0.5221408, 0.5562466, 0.5877288, 0.6185552, 0.6480698,
            0.6769285, 0.7051313, 0.5867320, 0.7320224, 0.7589135, 0.7851487,
        ])
        self.assertTrue(np.all(np.abs(selected - expected) < 0.0000001))

    def test_level(self):
        set1 = SelectionSet[ForestStand, ReferenceTrees](
            lambda _, trees: ((trees.breast_height_diameter > 10) &
                              (trees.management_category <= TreeManagementCategory.NO_RESTRICTION)),
            "breast_height_diameter",
            "stems_per_ha",
            TargetType.RELATIVE,
            0.5,
            np.array([0.0, 0.5, 1.0]),
            np.array([0.01, 0.5, 0.99]),
            ProfileXMode.RELATIVE
        )

        sets = [set1]

        target = SelectionTarget(TargetType.ABSOLUTE_REMAIN, "stems_per_ha", 100)

        selected = select_units(self.stand, self.trees, target, sets, "stems_per_ha", mode=Mode.LEVEL)
        expected = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.35888868, 0.81755099,
                             1.26578916, 1.13702738, 1.69317904, 2.12056893, 2.53753466])
        self.assertTrue(np.all(np.abs(selected - expected) < 0.0000001))

    def test_relative_remain(self):
        set1 = SelectionSet[ForestStand, ReferenceTrees](
            lambda _, trees: trees.management_category <= TreeManagementCategory.NO_RESTRICTION,
            "breast_height_diameter",
            "stems_per_ha",
            TargetType.RELATIVE,
            0.5,
            np.array([0.0, 0.5, 1.0]),
            np.array([0.01, 0.5, 0.99]),
            ProfileXMode.RELATIVE
        )

        sets = [set1]

        target = SelectionTarget(TargetType.RELATIVE_REMAIN, "stems_per_ha", 0.5)

        selected = select_units(self.stand, self.trees, target, sets, "stems_per_ha")
        expected = np.array([0.03403942, 0.62297545, 0.97687652, 1.32465831, 1.27813817,
                             1.56559083, 1.83683625, 2.10071426, 2.36219121, 2.6234402,
                             2.89303917, 3.15928607, 3.43787639, 3.72293374, 4.02080663,
                             4.33213228, 3.63666156, 4.64956199, 4.98923207, 5.34436415])
        self.assertTrue(np.all(np.abs(selected - expected) < 0.0000001))

    def test_multiple_sets(self):
        set1 = SelectionSet[ForestStand, ReferenceTrees](
            lambda _, trees: ((trees.breast_height_diameter > 10) &
                              (trees.management_category <= TreeManagementCategory.NO_RESTRICTION)),
            "breast_height_diameter",
            "stems_per_ha",
            TargetType.RELATIVE,
            0.2,
            np.array([0.0, 0.5, 1.0]),
            np.array([0.01, 0.5, 0.99]),
            ProfileXMode.RELATIVE
        )

        set2 = SelectionSet[ForestStand, ReferenceTrees](
            lambda _, trees: trees.management_category <= TreeManagementCategory.NO_RESTRICTION,
            "breast_height_diameter",
            "stems_per_ha",
            TargetType.RELATIVE,
            0.3,
            np.array([0.0, 1.0]),
            np.array([1.0, 0.5]),
            ProfileXMode.RELATIVE
        )

        sets = [set1, set2]

        target = SelectionTarget(TargetType.ABSOLUTE_REMAIN, "stems_per_ha", 50)

        selected = select_units(self.stand, self.trees, target, sets, "stems_per_ha")
        expected = np.array([7.04782, 3.03996227, 2.42840117, 2.16134462, 2.06740577,
                             1.81713722, 1.62741566, 1.48247347, 1.41888761, 1.38408961,
                             1.3765677, 1.39794721, 1.45419776, 1.55499185, 1.72233693,
                             1.99750541, 1.72617314, 2.45614463, 3.35445959, 5.45672])
        self.assertTrue(np.all(np.abs(selected - expected) < 0.000001))
