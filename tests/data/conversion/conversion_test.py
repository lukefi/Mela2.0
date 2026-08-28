from lukefi.metsi.data.conversion import vmi2internal
from lukefi.metsi.data.enums.internal import CuttingMethod
from lukefi.metsi.data.enums.vmi import VmiIteration
from lukefi.metsi.data.formats.nfi import vmi_util
from lukefi.metsi.data.formats.nfi.vmi12_builder import VMI12Builder
from lukefi.metsi.data.formats.nfi.vmi13_builder import VMI13Builder
from lukefi.metsi.data.formats.nfi.vmi_const import *
from lukefi.metsi.core.exceptions import MetsiException
from tests.data import test_util


class TestConversion(test_util.ConverterTestSuite):
    def test_determine_area_factors(self):
        assertions = [
            (['0', '0'], (0.0, 0.0)),
            (['0', '1'], (0.0, 0.1)),
            (['1', '0'], (0.1, 0.0)),
            (['2', None], (0.2, 0.0)),
            ([None, '4'], (0.0, 0.4)),
            ([None, None], (0.0, 0.0)),
            (['2', '4'], (0.2, 0.4)),
        ]

        self.run_with_test_assertions(assertions, vmi_util.determine_area_factors)

    def test_determine_artificial_regeneration_year(self):
        assertions = [
            (['1', '0', 2020], 2020),
            (['2', '0', 2020], 2020),
            (['3', '0', 2020], 2020),
            (['4', '0', 2020], 2020),
            (['4', '1', 2020], 2019),
            (['4', '2', 2020], 2017),
            (['4', '3', 2020], 2012),
            (['4', 'a', 2020], 2000),
            (['4', 'A', 2020], 2000),
            (['4', 'b', 2020], 1985),
            (['4', 'B', 2020], 1985),
            (['4', '.', 2020], None),
            (['', '', 2020], None)
        ]
        self.run_with_test_assertions(assertions, vmi_util.determine_artificial_regeneration_year)

    def test_determinate_clearing_of_reform_sector(self):
        assertions = [
            (['4', '-', 2020], None),
            (['4', '.', 2020], None),
            (['4', '0', 2020], 2020),
            (['4', '1', 2020], 2019),
            (['4', '2', 2020], 2017),
            (['4', '3', 2020], 2012),
            (['4', '4', 2020], None),
            (['1', '0', 2020], None),
            (['kissa123', '0', 2020], None)
        ]
        self.run_with_test_assertions(assertions, vmi_util.determine_clearing_of_reform_sector_year)

    def test_drainage_year(self):
        assertions = [
            (['10', 2020], 2010),
            (['10', None], None),
            (['abc', 2020], None),
            ([None, 2020], None)
        ]
        self.run_with_test_assertions(assertions, vmi_util.determine_drainage_year)

    def test_determine_vmi12_area_ha(self):
        self.assertRaises(IndexError, VMI12Builder._determine_area_ha, -1, -1)

        for county in range(1, 17):
            self.assertEqual(
                round(VMI12_COUNTY_AREAS[county - 1], 4),
                VMI12Builder._determine_area_ha(0, county))

        assertions = [
            ([2, 17], 0.0),
            ([3, 17], round(VMI12_COUNTY_AREAS[16], 4)),
            ([4, 17], round(VMI12_COUNTY_AREAS[17], 4)),
            ([2, 18], round(VMI12_COUNTY_AREAS[18], 4)),
            ([2, 19], 0.0),
            ([4, 19], round(VMI12_COUNTY_AREAS[19], 4)),
            ([5, 19], round(VMI12_COUNTY_AREAS[20], 4)),
            ([5, 20], 0.0),
            ([5, 21], round(VMI12_COUNTY_AREAS[21], 4)),
        ]
        self.run_with_test_assertions(assertions, VMI12Builder._determine_area_ha)

    def test_determine_vmi13_area_ha(self):
        self.assertRaises(IndexError, VMI13Builder._determine_area_ha, -1, -1, -1)
        self.assertRaises(Exception, VMI13Builder._determine_area_ha, 0, 0, 0)
        self.assertRaises(Exception, VMI13Builder._determine_area_ha, 3, 0, 0)
        self.assertRaises(Exception, VMI13Builder._determine_area_ha, 5, 2, 1)
        self.assertRaises(Exception, VMI13Builder._determine_area_ha, 5, 2, None)

        assertions = [
            ([1, 2, 0], 345.73918),
            ([2, 2, 0], 338.0386443),
            ([4, 2, 0], 342.975010960105),
        ]
        self.run_with_test_assertions(assertions, VMI13Builder._determine_area_ha)

    def test_tax_class_reduction(self):
        assertions = [
            (['0'], 0),
            (['1'], 1),
            (['2'], 2),
            (['3'], 3),
            (['4'], 4),
            (['xxxx'], 0)
        ]
        self.run_with_test_assertions(assertions, vmi_util.determine_tax_class_reduction)

    def test_tax_class(self):
        assertions = [
            (['0'], 1),
            (['1'], 2),
            (['2'], 3),
            (['3'], 4),
            (['4'], 5),
            (['xxxx'], 0)
        ]
        self.run_with_test_assertions(assertions, vmi_util.determine_tax_class)

    def test_vmi12_height_above_sea_level(self):
        assertions = [
            (['1233'], 123.3),
            ([' '], None)
        ]
        self.run_with_test_assertions(assertions, VMI12Builder._transform_height_above_sea_level)

    def test_vmi_degree_days(self):
        assertions = [
            (['1233'], 1233.0),
            (['121.2'], 121.2),
            ([' '], None)
        ]
        self.run_with_test_assertions(assertions, vmi_util.transform_vmi_degree_days)

    def test_soil_surface_preparation_year(self):
        assertions = [
            (['0', 1779], 1779),
            (['1', 1779], 1778),
            (['2', 1779], 1776),
            (['3', 1779], 1771),
            (['A', 1779], 1759),
            (['a', 1779], 1759),
            (['xxxx', 1779], None)
        ]
        self.run_with_test_assertions(assertions, vmi_util.determine_soil_surface_preparation_year)

    def test_forest_maintenance_details(self):
        assertions = [
            (['0', '2', 2020], (None, None, None)),
            (['1', '2', 2020], (2018, None, None)),
            (['2', '3', 2020], (2017, None, None)),
            (['3', '6', 2020], (None, 2013, CuttingMethod.FIRST_THINNING)),
            (['8', 'A', 2020], (None, 2000, CuttingMethod.SEED_TREE_CUTTING)),
        ]
        self.run_with_test_assertions(assertions, vmi2internal.convert_forest_maintenance_details)

    def test_forest_maintenance_year(self):
        assertions = [
            (['0', 2020], 2020),
            (['1', 2020], 2019),
            (['2', 2020], 2018),
            (['3', 2020], 2017),
            (['4', 2020], 2016),
            (['5', 2020], 2015),
            (['6', 2020], 2013),
            (['a', 2020], 2000),
            (['A', 2020], 2000),
            (['b', 2020], 1980),
            (['B', 2020], 1980),
            (['', 2020], None)
        ]
        self.run_with_test_assertions(assertions, vmi2internal._determine_forest_maintenance_year)
        self.assertRaises(ValueError, vmi2internal._determine_forest_maintenance_year, 'kissa123', None)
        self.assertRaises(ValueError, vmi2internal._determine_forest_maintenance_year, 'kissa123', 2020)
        self.assertRaises(AttributeError, vmi2internal._determine_forest_maintenance_year, None, 2020)

    def test_forest_maintenance_method(self):
        assertions = [
            (['0', 2007], CuttingMethod.NO_CUTTING),
            (['4', 2007], CuttingMethod.THINNING),
            (['7', 2007], CuttingMethod.CLEARCUTTING),
            (['3', 2007], CuttingMethod.FIRST_THINNING),
            (['6', 2007], CuttingMethod.OVER_STORY_REMOVAL),
            (['8', 2007], CuttingMethod.SEED_TREE_CUTTING),
            (['9', 2007], CuttingMethod.SHELTERWOOD_CUTTING),
            (['9', -2007], CuttingMethod.NO_CUTTING),
            (['0', None], CuttingMethod.NO_CUTTING),
            (['4', None], CuttingMethod.NO_CUTTING),
        ]

        self.run_with_test_assertions(assertions, vmi2internal._convert_cutting_method)

    def test_vmi12_date(self):
        source = '010219'
        result = vmi_util.parse_date(source)
        self.assertEqual(result.year, 2019)
        self.assertEqual(result.month, 2)
        self.assertEqual(result.day, 1)

    def test_vmi13_date(self):
        source = '20190201'
        result = VMI13Builder._parse_date(source)
        self.assertEqual(result.year, 2019)
        self.assertEqual(result.month, 2)
        self.assertEqual(result.day, 1)

    def test_vmi12_dominant_storey_age(self):
        assertions = [
            (("10.0", "2.0"), (12.0))
        ]
        self.run_with_test_assertions(assertions, VMI12Builder._determine_dominant_storey_age)

    def test_parse_forestry_centre(self):
        assertions = [
            (['20'], 20),
            (['10'], 10),
            (['kissa123'], 10),
            ([''], 10),
            ([None], 10)
        ]
        self.run_with_test_assertions(assertions, vmi_util.parse_forestry_centre)

    def test_determine_municipality(self):
        assertions = [
            (['  1', '  2'], 1),
            (['1', '  2'], 1),
            (['.', '  2'], 2),
            (['.', '2'], 2),
            (['.', '.'], None),
            (['kissa123', '.'], None),
            (['.', 'kettu456'], None)
        ]
        self.run_with_test_assertions(assertions, vmi_util.determine_municipality)

    def test_vmi_codevalue(self):
        assertions = [
            (['abc'], 'abc'),
            (['2'], '2'),
            (['  2'], '2'),
            (['   '], None),
            ([''], None),
            (['.'], None)
        ]
        self.run_with_test_assertions(assertions, vmi_util.vmi_codevalue)

    def test_determine_tree_height(self):
        assertions = [
            (['111', 10.0], 11.1),
            (['111', 100.0], 1.11),
        ]
        self.run_with_test_assertions(assertions, vmi_util.determine_tree_height)

    def test_transform_tree_diameter(self):
        assertions = [
            (['123.4'], 12.34),
            (['.'], 0.0),
            ([None], 0.0)
        ]
        self.run_with_test_assertions(assertions, vmi_util.transform_tree_diameter)

    def test_determine_stems_per_ha(self):
        assertions = [
            # VMI13 parameters
            ([3.0, VmiIteration.VMI13, None, None], 2122.06591),
            ([6.0, VmiIteration.VMI13, None, None], 198.94368),
            ([12.0, VmiIteration.VMI13, None, None], 39.29752),

            # VMI12 parameters (different r1 => different mid-diameter plateau)
            ([6.0, VmiIteration.VMI12, None, None], 100.06724),
            ([12.0, VmiIteration.VMI12, None, None], 39.29752),

            # Zero/negative diameter always => 1.0
            ([0.0, VmiIteration.VMI13, None, None], 1.0),
            ([0.0, VmiIteration.VMI12, None, None], 1.0),

            # VMI11 special-case Ahvenanmaa (metkes=0, ahvkeilaus='A')
            ([12.0, VmiIteration.VMI11, 0, "A"], 88.41941),

            ([30.0, VmiIteration.VMI10, 1, None], 28.29421),   # south
            ([30.0, VmiIteration.VMI10, 12, None], 21.22066),  # north
        ]
        self.run_with_test_assertions(assertions, vmi_util.determine_stems_per_ha)

        # Unsupported VMI version should raise
        self.assertRaises(MetsiException, vmi_util.determine_stems_per_ha, 10.0, 99, None, None)

    def test_determine_tree_age_values(self):
        assertions = [
            (['0', '0', '0'], (None, None)),
            (['10', '0', '0'], (10, 19)),
            (['10', '2', '0'], (10, 12)),
            (['10', '2', '23'], (10, 23)),
            (['10', '2', '3'], (10, 3)),
            (['10', '2', ''], (10, 12)),
            (['10', '', ''], (10, 19)),
            (['', '', ''], (None, None)),
            (['', '', '34'], (None, 34)),
            (['', '23', ''], (None, None)),
        ]
        self.run_with_test_assertions(assertions, vmi_util.determine_tree_age_values)

    def test_determine_tree_management_category(self):
        assertions = [
            (['A'], 1),
            (['a'], 1),
            (['B'], 2),
            (['b'], 2),
            (['C'], 2),
            (['c'], 2),
            (['D'], 2),
            (['d'], 2),
            (['E'], 2),
            (['e'], 2),
            (['F'], 2),
            (['f'], 2),
            (['G'], 2),
            (['g'], 2),
            (['H'], 1),
            (['kissa123'], 1),
            ([''], 1),
        ]
        self.run_with_test_assertions(assertions, vmi_util.determine_tree_management_category)

    def test_determine_stratum_tree_height(self):
        assertions = [
            (['10'], 1.0),
            (['.'], None),
            ([None], None),
            (['kissa123'], None),
        ]
        self.run_with_test_assertions(assertions, vmi_util.determine_stratum_tree_height)

    def test_determine_stratum_origin(self):
        assertions = [
            (['0'], 0),
            (['1'], 0),
            (['2'], 0),
            (['3'], 2),
            (['4'], 1),
            ([""], None),
            ([' '], None),
        ]
        self.run_with_test_assertions(assertions, vmi2internal.convert_origin)

    def test_determine_stratum_age_values(self):
        assertions = [
            (['0', '0', 0], (0.0, 0.0)),
            (['0', '1', 0], (10.0, 1.0)),
            (['2', '1', 0], (3.0, 1.0)),
            (['0', '0', 1.0], (1.0, 0.0)),
            (['0', '0', 10.0], (22.0, 14.0)),
            (['1', '0', 10.0], (15.0, 14.0)),
            (['1', '1', 0.0], (2.0, 1.0)),
            (['.', ' ', 0.0], (0.0, 0.0)),
            (['.', ' ', 10.0], (22.0, 14.0)),
        ]
        self.run_with_test_assertions(assertions, vmi_util.determine_stratum_age_values)

    def test_generating_vmi12_stand_identifier(self):
        # section_x is 012
        # section_y is 001
        # test_area_number is 01
        # stand_number is 1
        stand = vmi_util.generate_source_data(VMI12_STAND_INDICES, 'K0001012 01 11    66569963102608    1010   0041721         000059500417      1   0         40020618 B0          0   0 0   0              0  0                   6654199.85 C 102600.11 66569963102608                                                                                          0      0')
        stand_id = vmi_util.generate_stand_identifier(stand)
        self.assertEqual('0-001-012-01-1', stand_id)

    def test_generating_vmi12_stratum_identifier(self):
        # section_x is 023
        # section_y is 002
        # test_area_number is 02
        # stand_number is 1
        stratum = vmi_util.generate_source_data(VMI12_STRATUM_INDICES, 'K0002023 02 12 01  1 11             24 190  04606N17 1  84A1 0')
        stratum_id = vmi_util.generate_stratum_identifier(stratum)
        self.assertEqual('0-002-023-02-1-01-stratum', stratum_id)

    def test_generating_vmi12_tree_identifier(self):
        # section_x is 023
        # section_y is 002
        # test_area_number is 02
        # stand_number is 1
        tree = vmi_util.generate_source_data(VMI12_TREE_INDICES, 'K0002023 02 13001 1207217 2  01 15521741  020081711 00                                                              7725 3999  342                                                                                                                                           259959  134571   11515 39185 101864  4303 11769  8489 24696 4196')
        tree_id = vmi_util.generate_tree_identifier(tree)
        self.assertEqual('0-002-023-02-1-001-tree', tree_id)

    def test_generating_vmi13_stand_identifier(self):
        # section_x is 75
        # section_y is 58
        # test_area_number is 10
        # stand_number is 1
        vmi13_row = VMI13Builder._generate_source_data(VMI13_STAND_INDICES, '1 U 1  58  75 10 1   . 0 20200522 2020 258 3 1 10 10  . 12 10 176 176 893    1    5 4 S 7025056.83 589991.91 7025054.56 589991.44  179.70 1019    . T  1 3   33 220  0   . 0  . 1 0  0  . . 0 1  0  . . 0 0 2 3 0  . 2 3 1 35 2 3 2 0 4  75 0 0 3 1 5  2 .  . .  . 15 4 10 0 15 2 10 8 15 6 26 .  .  .    . 22 187  63 19 . U     . E 1 . . 0 A . . . . 0 . 0 .  . 0 . 7 3 . . 4 1 . 2 2 2   0 . . 0 . .   1  0 0 .   . 0 0 . . .         . 1 7025054.56 589991.44 .    .')
        vmi13_id = vmi_util.generate_stand_identifier(vmi13_row)
        self.assertEqual('1-58-75-10-1', vmi13_id)

    def test_generating_vmi13_stratum_identifier(self):
        # section_x is 68
        # section_y is 59
        # test_area_number is 2
        # stand_number is 1
        stratum = VMI13Builder._generate_source_data(VMI13_STRATUM_INDICES, '2 U 1  59  68  2 1   1 0 20200503 258 1  2 3 1350  1400  4  38 E   7  8 F  2 .  0 .  .  . .  . .    .')
        stratum_id = vmi_util.generate_stratum_identifier(stratum)
        self.assertEqual('1-59-68-2-1-1-stratum', stratum_id)

    def test_generating_vmi13_tree_identifier(self):
        # section_x is 75
        # section_y is 58
        # test_area_number is 10
        # stand_number is 1
        tree = VMI13Builder._generate_source_data(VMI13_TREE_INDICES, '3 U 1  58  75 10 1  10 0 20200522 258  11 V  1  250 7 2    .    . 306  863 1  0 0 .   .   .   .   .  . . .  .  .  .  . .  . .  . . . . . .   .   . .  .   .   .   .   . .   . . .   . . .   . . .   . . .   . . .   . . .   . . .   . . . .  . .  . .  . .  . .  . .  . .  . .  .     .     .     .     .     .     .     .     .     .        .        .        .     .       .      .      .      .      .     . .    .')
        tree_id = vmi_util.generate_tree_identifier(tree)
        self.assertEqual('1-58-75-10-1-10-tree', tree_id)
