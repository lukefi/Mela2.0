import unittest
import numpy as np
import pandas as pd
from lukefi.metsi.data.enums.internal import (
    DrainageCategory,
    LandUseCategory,
    OwnerCategory,
    SiteType,
    SoilPeatlandCategory)
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.util.select_units import SelectionSet, SelectionTarget, select_units
from lukefi.metsi.data.vector_model import ReferenceTrees


class TestSelectTrees(unittest.TestCase):

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
        self.stand.drainage_feasibility = bool(0)
        self.stand.drainage_year = 0
        self.stand.fertilization_year = 0
        self.stand.soil_surface_preparation_year = 2023
        self.stand.natural_regeneration_feasibility = bool(0)
        self.stand.regeneration_area_cleaning_year = 0
        self.stand.young_stand_tending_year = 0
        self.stand.pruning_year = 0
        self.stand.cutting_year = 0
        self.stand.forest_management_category = 1
        self.stand.method_of_last_cutting = 0
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
        self.trees.origin = np.repeat(0, self.trees.size)
        self.trees.management_category = np.ascontiguousarray(pd_trees.manag_cat, np.int32)
        self.trees.saw_log_volume_reduction_factor = np.repeat(0.0, self.trees.size)
        self.trees.pruning_year = np.repeat(0, self.trees.size)
        self.trees.age_when_10cm_diameter_at_breast_height = np.repeat(0, self.trees.size)
        self.trees.stand_origin_relative_position = np.repeat(np.asarray([[0.0, 0.0, 0.0]]), self.trees.size, axis=0)
        self.trees.lowest_living_branch_height = np.repeat(0.0, self.trees.size)
        self.trees.tree_category = np.repeat("", self.trees.size)
        self.trees.storey = np.repeat(0, self.trees.size)
        self.trees.sapling = np.repeat(False, self.trees.size)
        self.trees.tree_type = np.repeat("", self.trees.size)
        self.trees.tuhon_ilmiasu = np.repeat("", self.trees.size)
        self.trees.latvuskerros = np.repeat(0.0, self.trees.size)
        self.stand.reference_trees = self.trees

    def test_odds_units(self):
        set3 = SelectionSet[ForestStand, ReferenceTrees]()
        set3.sfunction = lambda _, trees: trees.management_category <= 1
        set3.order_var = "breast_height_diameter"
        set3.target_var = "stems_per_ha"
        set3.target_type = "relative"
        set3.target_amount = 0.5
        set3.profile_x = np.array([0.0, 0.5, 1.0])
        set3.profile_y = np.array([0.01, 0.5, 0.99])
        set3.profile_xmode = "relative"

        sets = [set3]

        target = SelectionTarget()
        target.type = "absolute_remain"
        target.var = "stems_per_ha"
        target.amount = 100

        selected = select_units(self.stand, self.trees, target, sets)
        expected = np.array([0.001285986, 0.026337014, 0.044398658, 0.064245866, 0.062085809, 0.081328615,
                             0.102171400, 0.125488095, 0.152262229, 0.183603001, 0.222128271, 0.268294736,
                             0.328393045, 0.407688528, 0.519838362, 0.690429783, 0.617979504, 0.971546129,
                             1.562742457, 3.499975853])
        self.assertTrue(np.all(np.abs(selected - expected) < 0.000000001))

    def test_odds_profile(self):
        set3 = SelectionSet[ForestStand, ReferenceTrees]()
        set3.sfunction = lambda _, trees: trees.management_category <= 1
        set3.order_var = "breast_height_diameter"
        set3.target_var = "stems_per_ha"
        set3.target_type = "relative"
        set3.target_amount = 0.5
        set3.profile_x = np.array([0.0, 0.5, 1.0])
        set3.profile_y = np.array([0.01, 0.5, 0.99])
        set3.profile_xmode = "relative"

        sets = [set3]

        target = SelectionTarget()
        target.type = "absolute_remain"
        target.var = "stems_per_ha"
        target.amount = 100

        selected = select_units(self.stand, self.trees, target, sets, mode="odds_profile")
        expected = np.array([0.0002173194, 0.0069249673, 0.0103028322, 0.0133604573, 0.0128780946,
                             0.0151229730, 0.0514119379, 0.1843021961, 0.3066456085, 0.4205515442,
                             0.5302387415, 0.6314884621, 0.7306288135, 0.8255504266, 0.9183626704,
                             1.0090655451, 0.8502398744, 1.0955496815, 1.1820338178, 1.2664085850,])
        self.assertTrue(np.all(np.abs(selected - expected) < 0.000000001))

    def test_scale(self):
        set3 = SelectionSet[ForestStand, ReferenceTrees]()
        set3.sfunction = lambda _, trees: trees.management_category <= 1
        set3.order_var = "breast_height_diameter"
        set3.target_var = "stems_per_ha"
        set3.target_type = "relative"
        set3.target_amount = 0.5
        set3.profile_x = np.array([0.0, 0.5, 1.0])
        set3.profile_y = np.array([0.01, 0.5, 0.99])
        set3.profile_xmode = "relative"

        sets = [set3]

        target = SelectionTarget()
        target.type = "absolute_remain"
        target.var = "stems_per_ha"
        target.amount = 100

        selected = select_units(self.stand, self.trees, target, sets, mode="scale")
        expected = np.array([
            0.0102433, 0.1676316, 0.2474976, 0.3199432, 0.3083750, 0.3614502, 0.4073618,
            0.4486822, 0.4867233, 0.5221408, 0.5562466, 0.5877288, 0.6185552, 0.6480698,
            0.6769285, 0.7051313, 0.5867320, 0.7320224, 0.7589135, 0.7851487,
        ])
        self.assertTrue(np.all(np.abs(selected - expected) < 0.0000001))

    def test_level(self):
        set3 = SelectionSet[ForestStand, ReferenceTrees]()
        set3.sfunction = lambda _, trees: trees.management_category <= 1
        set3.order_var = "breast_height_diameter"
        set3.target_var = "stems_per_ha"
        set3.target_type = "relative"
        set3.target_amount = 0.5
        set3.profile_x = np.array([0.0, 0.5, 1.0])
        set3.profile_y = np.array([0.01, 0.5, 0.99])
        set3.profile_xmode = "relative"

        sets = [set3]

        target = SelectionTarget()
        target.type = "absolute_remain"
        target.var = "stems_per_ha"
        target.amount = 100

        selected = select_units(self.stand, self.trees, target, sets, mode="level")
        expected = np.array([
            0.0704782, 1.1533761, 1.7028871, 2.2013436, 2.1217496, 2.4869289, 2.8028201,
            3.0871221, 3.3488604, 3.5925479, 3.8272099, 4.0438209, 4.2559193, 4.4589921,
            4.6575523, 4.8515997, 4.0369623, 5.0366216, 5.2216436, 5.4021528,
        ])
        self.assertTrue(np.all(np.abs(selected - expected) < 0.0000001))

    def test_relative_remain(self):
        set3 = SelectionSet[ForestStand, ReferenceTrees]()
        set3.sfunction = lambda _, trees: trees.management_category <= 1
        set3.order_var = "breast_height_diameter"
        set3.target_var = "stems_per_ha"
        set3.target_type = "relative"
        set3.target_amount = 0.5
        set3.profile_x = np.array([0.0, 0.5, 1.0])
        set3.profile_y = np.array([0.01, 0.5, 0.99])
        set3.profile_xmode = "relative"

        sets = [set3]

        target = SelectionTarget()
        target.type = "relative_remain"
        target.var = "stems_per_ha"
        target.amount = 0.5

        selected = select_units(self.stand, self.trees, target, sets)
        expected = np.array([
            0.03403949, 0.62297660, 0.97687819, 1.32466043, 1.27814020, 1.56559315,
            1.83683878, 2.10071695, 2.36219400, 2.62344304, 2.89304200, 3.15928884,
            3.43787904, 3.72293620, 4.02080883, 4.33213414, 3.63666300, 4.64956342,
            4.98923296, 5.34436438,
        ])
        self.assertTrue(np.all(np.abs(selected - expected) < 0.0000001))

    def test_multiple_sets(self):
        set1 = SelectionSet[ForestStand, ReferenceTrees]()
        set1.sfunction = lambda _, trees: (trees.breast_height_diameter > 10) & (trees.management_category <= 1)
        set1.order_var = "breast_height_diameter"
        set1.target_var = "stems_per_ha"
        set1.target_type = "relative"
        set1.target_amount = 0.2
        set1.profile_x = np.array([0.0, 0.5, 1.0])
        set1.profile_y = np.array([0.01, 0.5, 0.99])
        set1.profile_xmode = "relative"

        set2 = SelectionSet[ForestStand, ReferenceTrees]()
        set2.sfunction = lambda _, trees: trees.management_category <= 1
        set2.order_var = "breast_height_diameter"
        set2.target_var = "stems_per_ha"
        set2.target_type = "relative"
        set2.target_amount = 0.3
        set2.profile_x = np.array([0.0, 1.0])
        set2.profile_y = np.array([1.0, 0.5])
        set2.profile_xmode = "relative"

        sets = [set1, set2]

        target = SelectionTarget()
        target.type = "absolute_remain"
        target.var = "stems_per_ha"
        target.amount = 50

        selected = select_units(self.stand, self.trees, target, sets)
        expected = np.array([
            7.047820, 3.039971, 2.428410, 2.161354, 2.067414, 1.817145, 1.627423, 1.482481,
            1.418895, 1.384098, 1.376576, 1.397957, 1.454208, 1.555003, 1.722350, 1.997521,
            1.726186, 2.456163, 3.354480, 5.456720,
        ])
        self.assertTrue(np.all(np.abs(selected - expected) < 0.000001))
