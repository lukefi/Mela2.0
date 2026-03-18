from enum import Enum, IntEnum, StrEnum


class VmiIteration(StrEnum):
    VMI9 = "vmi9"
    VMI10 = "vmi10"
    VMI11 = "vmi11"
    VMI12 = "vmi12"
    VMI13 = "vmi13"


class VmiSpecies(Enum):
    TREELESS = "0"
    PINE = "1"
    SPRUCE = "2"
    SILVER_BIRCH = "3"
    DOWNY_BIRCH = "4"
    ASPEN = "5"
    GREY_ALDER = "6"
    COMMON_ALDER = "7"
    MOUNTAIN_ASH = "8"
    GOAT_WILLOW = "9"
    OTHER_CONIFEROUS = "A0"
    SHORE_PINE = "A1"
    KEDAR = "A2"
    OTHER_PINE = "A3"
    LARCH = "A4"
    ABIES = "A5"
    OTHER_SPRUCE = "A6"
    THUJA = "A7"
    JUNIPER = "A8"
    YEW = "A9"
    OTHER_DECIDUOUS = "B0"
    BAY_WILLOW = "B1"
    EUROPEAN_WHITE_ELM = "B2"
    WYCH_ELM = "B3"
    SMALL_LEAVED_LIME = "B4"
    POPLAR = "B5"
    COMMON_ASH = "B6"
    OAK = "B7"
    BIRD_CHERRY = "B8"
    MAPLE = "B9"
    HAZEL = "C1"
    UNKNOWN = None

    @classmethod
    def _missing_(cls, value):
        if value == "0":
            return cls.UNKNOWN
        return None


class VmiSpeciesNumeric(IntEnum):
    TREELESS = 10
    PINE = 1
    SPRUCE = 2
    SILVER_BIRCH = 3
    DOWNY_BIRCH = 4
    ASPEN = 5
    GREY_ALDER = 6
    COMMON_ALDER = 7
    MOUNTAIN_ASH = 8
    GOAT_WILLOW = 9
    OTHER_CONIFEROUS = 20
    SHORE_PINE = 11
    KEDAR = 12
    OTHER_PINE = 13
    LARCH = 14
    ABIES = 15
    OTHER_SPRUCE = 16
    THUJA = 17
    JUNIPER = 18
    YEW = 19
    OTHER_DECIDUOUS = 30
    BAY_WILLOW = 21
    EUROPEAN_WHITE_ELM = 22
    WYCH_ELM = 23
    SMALL_LEAVED_LIME = 24
    POPLAR = 25
    COMMON_ASH = 26
    OAK = 27
    BIRD_CHERRY = 28
    MAPLE = 29
    HAZEL = 31


_VMI_NUMERIC_TO_SPECIES_MAP = {
    VmiSpeciesNumeric.TREELESS: VmiSpecies.TREELESS,
    VmiSpeciesNumeric.PINE: VmiSpecies.PINE,
    VmiSpeciesNumeric.SPRUCE: VmiSpecies.SPRUCE,
    VmiSpeciesNumeric.SILVER_BIRCH: VmiSpecies.SILVER_BIRCH,
    VmiSpeciesNumeric.DOWNY_BIRCH: VmiSpecies.DOWNY_BIRCH,
    VmiSpeciesNumeric.ASPEN: VmiSpecies.ASPEN,
    VmiSpeciesNumeric.GREY_ALDER: VmiSpecies.GREY_ALDER,
    VmiSpeciesNumeric.COMMON_ALDER: VmiSpecies.COMMON_ALDER,
    VmiSpeciesNumeric.MOUNTAIN_ASH: VmiSpecies.MOUNTAIN_ASH,
    VmiSpeciesNumeric.GOAT_WILLOW: VmiSpecies.GOAT_WILLOW,
    VmiSpeciesNumeric.OTHER_CONIFEROUS: VmiSpecies.OTHER_CONIFEROUS,
    VmiSpeciesNumeric.SHORE_PINE: VmiSpecies.SHORE_PINE,
    VmiSpeciesNumeric.KEDAR: VmiSpecies.KEDAR,
    VmiSpeciesNumeric.OTHER_PINE: VmiSpecies.OTHER_PINE,
    VmiSpeciesNumeric.LARCH: VmiSpecies.LARCH,
    VmiSpeciesNumeric.ABIES: VmiSpecies.ABIES,
    VmiSpeciesNumeric.OTHER_SPRUCE: VmiSpecies.OTHER_SPRUCE,
    VmiSpeciesNumeric.THUJA: VmiSpecies.THUJA,
    VmiSpeciesNumeric.JUNIPER: VmiSpecies.JUNIPER,
    VmiSpeciesNumeric.YEW: VmiSpecies.YEW,
    VmiSpeciesNumeric.OTHER_DECIDUOUS: VmiSpecies.OTHER_DECIDUOUS,
    VmiSpeciesNumeric.BAY_WILLOW: VmiSpecies.BAY_WILLOW,
    VmiSpeciesNumeric.EUROPEAN_WHITE_ELM: VmiSpecies.EUROPEAN_WHITE_ELM,
    VmiSpeciesNumeric.WYCH_ELM: VmiSpecies.WYCH_ELM,
    VmiSpeciesNumeric.SMALL_LEAVED_LIME: VmiSpecies.SMALL_LEAVED_LIME,
    VmiSpeciesNumeric.POPLAR: VmiSpecies.POPLAR,
    VmiSpeciesNumeric.COMMON_ASH: VmiSpecies.COMMON_ASH,
    VmiSpeciesNumeric.OAK: VmiSpecies.OAK,
    VmiSpeciesNumeric.BIRD_CHERRY: VmiSpecies.BIRD_CHERRY,
    VmiSpeciesNumeric.MAPLE: VmiSpecies.MAPLE,
    VmiSpeciesNumeric.HAZEL: VmiSpecies.HAZEL,
}


class VmiFraLandUseClass(Enum):
    FOREST = "1"
    OTHER_WOODED_LAND = "2"
    OTHER_LAND = "3"
    OTHER_LAND_WITH_TREE_COVER = "4"


class VmiLandUseCategory(Enum):
    FOREST = '1'
    SCRUB_LAND = '2'
    WASTE_LAND = '3'
    OTHER_FOREST = '4'
    AGRICULTURAL = '5'
    BUILT_LAND = '6'
    ROAD = '7'
    ENERGY_TRANSMISSION_LINE = '8'
    FRESHWATER = 'A'
    SEA = 'B'
    OBSOLETE = 'C'


class VmiLandUseCategoryDetail(Enum):
    pass


class VmiProductiveForestLandDetail(VmiLandUseCategoryDetail):
    NORMAL = "0"
    PASTURE = "5"
    CONVERTED = "6"


class VmiPoorlyProductiveForestLandDetail(VmiLandUseCategoryDetail):
    NORMAL = "0"
    PASTURE = "5"
    CONVERTED = "6"


class VmiUnproductiveLandDetail(VmiLandUseCategoryDetail):
    VEGETATION_COVER = "0"
    NO_VEGETATION_COVER = "1"


class VmiOtherForestryLandDetail(VmiLandUseCategoryDetail):
    OTHER_MINERAL_SOIL = "1"
    OTHER_ORGANIC_SOIL = "2"
    SEED_PRODUCTION_FOREST = "4"
    FORESTRY_BUILDINGS = "6"
    FOREST_ROAD = "7"
    SMALL_SCALE_PEAT_PRODUCTION = "8"
    GRAVEL_OR_SAND_PRODUCTION = "9"


class VmiArableLandDetail(VmiLandUseCategoryDetail):
    FARMED_FIELDS_OR_FALLOWS = "0"
    ABANDONED_FIELD_MINERAL_SOIL = "1"
    ABANDONED_FIELD_ORGANIC_SOIL = "2"
    ABANDONED_FIELD_REFORESTING_MINERAL_SOIL = "3"
    ABANDONED_FIELD_REFORESTING_ORGANIC_SOIL = "4"
    PASTURE_MEADOW = "5"
    BIOENERGY_PRODUCTION_WOODY_PLANTS_MINERAL_SOIL = "6"
    BIOENERGY_PRODUCTION_WOODY_PLANTS_ORGANIC_SOIL = "7"
    BIOENERGY_PRODUCTION_NON_WOODY_PLANTS_MINERAL_SOIL = "8"
    BIOENERGY_PRODUCTION_NON_WOODY_PLANTS_ORGANIC_SOIL = "9"
    UNPRODUCTIVE_LAND = "A"
    BARNS_AND_OTHER_AGRICULTURAL_BUILDINGS = "B"
    FRUIT_TREE_OR_BERRY_SHRUB_PLANTATION = "C"


class VmiBuildUpLandDetail(VmiLandUseCategoryDetail):
    PEAT_PRODUCTION_PREPARATION = "1"
    PEAT_PRODUCTION_OUT_OF_USE = "2"
    PEAT_PRODUCTION_MAINTENANCE = "3"
    SURFACE_DRAINAGE_ON_PEAT_PRODUCTION = "4"
    GREEN_HOUSE_YARD_HOME_GARDEN = "5"
    MINING_AREA = "6"
    PEAT_PRODUCTION_ONGOING = "8"
    GRAVEL_OR_SAND_PRODUCTION = "9"
    OTHER_BUILD_UP_LAND = "0"


class VmiInlandWaterDetail(VmiLandUseCategoryDetail):
    NATURAL_WATER_BASIN = "0"
    ARTIFICIAL_LAKE_OR_TAMED_RIVER = "8"


class VmiOwnerCategory(Enum):
    UNKNOWN = "0"
    # private
    PRIVATE = "1"
    # enterprise
    FOREST_INDUSTRY_ENTERPRISE = "2"
    OTHER_ENTERPRISE = "3"
    # state forest
    METSAHALLITUS = "4"
    OTHER_STATE_AGENCY = "5"
    # communities
    FOREST_COOP = "6"  # = yhteismetsä
    MUNICIPALITY = "7"
    CONGREGATION = "8"
    OTHER_COMMUNITY = "9"
    # jakamaton
    UNDIVIDED = "A"  # = jakamaton kuolinpesä


class VmiSoilPeatlandCategory(Enum):
    MINERAL_SOIL = '1'
    SPRUCE_MIRE = '2'
    PINE_MIRE = '3'
    TREELESS_MIRE = '4'


class VmiSiteType(Enum):
    LEHTO = '1'
    LEHTOMAINEN_KANGAS = '2'
    TUOREKANGAS = '3'
    KUIVAHKOKANGAS = '4'
    KUIVAKANGAS = '5'
    KARUKKOKANGAS = '6'
    KALLIOMAA_TAI_HIETIKKO = '7'
    LAKIMETSA_TAI_TUNTURIHAVUMETSA = '8'
    TUNTURIKOIVIKKO = 'T'
    AVOTUNTURI = 'A'


class VmiDrainageCategory(Enum):
    OJITTAMATON_KANGAS_TAI_SUO = '0'
    OJITETTU_KANGAS = '1'
    OJIKKO = '2'
    MUUTTUMA = '3'
    TURVEKANGAS = '4'


class VmiStratumRank(Enum):
    UNPRODUCTIVE_SEEDLINGS = '0'
    DOMINANT_STOREY = '1'
    OVER_STOREY = '2'
    RETENTION_TREE_STOREY = '3'
    NURSE_CROP = '4'
    UNDER_STOREY_CAPABLE_FOR_DEVELOPMENT = '5'
    UNDER_STOREY_NOT_CAPABLE_FOR_DEVELOPMENT = '6'
    NON_ESTABLISHED_SEEDLINGS = '7'
    DAMAGED_TREE_STRATUM = '8'
    SEEDLING_STRATUM = '9'


class VmiTreeStorey(Enum):
    DOMINANT_MAIN = '2'
    DOMINANT_MIDDLE = '3'
    DOMINANT_LOWER = '4'
    UNDER = '5'
    OVER_MAIN = '6'
    OVER_OTHER = '7'
    DOMINANT_SPARE_1 = 'B'
    DOMINANT_SPARE_2 = 'C'
    DOMINANT_SPARE_3 = 'D'
    UNDER_SPARE_1 = 'E'
    OVER_SPARE_1 = 'F'
    OVER_SPARE_2 = 'G'


class VmiOrigin(Enum):
    UNKNOWN = '0'
    NATURAL_SEED = '1'
    NATURAL_SPROUT = '2'
    PLANTED = '3'
    SEEDED = '4'


def convert_vmi_numeric_to_species(numeric: VmiSpeciesNumeric) -> VmiSpecies:
    return _VMI_NUMERIC_TO_SPECIES_MAP[numeric]


class VmiTreeType(Enum):
    REMEASURED_TALLY_TREE = 'V'
    NEW_TALLY_TREE_INCREMENT_HEIGHT_GREATER_THAN_1_3_M = 'U'
    NEW_TALLY_TREE_INCREMENT_HEIGHT_LESS_THAN_1_3_M = 'S'
    NEW_TALLY_TREE_OTHER_THAN_INCREMENT = 'T'
    OLD_TALLY_TREE_STUMP_STEM_REMOVED = 'K'
    OLD_TALLY_TREE_STUMP_STEM_NOT_REMOVED = 'R'
    OLD_TALLY_TREE_MEASURED_PREVIOUSLY_BY_MISTAKE = 'N'
    OLD_TALLY_TREE_MEASURED_PREVIOUSLY_BY_MISTAKE_NO_LONGER_TALLY = 'Z'
    OLD_TALLY_TREE_LAND_USE_CLASS_CHANGED_NO_LONGER_EXISTS = 'M'
    OLD_TALLY_TREE_LAND_USE_CLASS_CHANGED_STILL_EXISTS = 'J'


class VmiTreeCategory(Enum):
    SMALL_TREE = '0'
    WASTE_TREE = '1'
    PULP_WOOD_TREE = '3'
    SAW_LOG_TREE = '7'
    USABLE_STANDING_DEAD_TREE = 'A'
    USABLE_FALLEN_DEAD_TREE = 'B'
    UNUSABLE_DEAD_TREE = 'D'
    STUMP_ALIVE_WHEN_FELLING = 'E'
    STUMP_DEAD_STANDING_WHEN_FELLING = 'F'
    STUMP_DEAD_FALLEN_WHEN_FELLING = 'G'


class VmiDamageType(Enum):
    NO_DAMAGE = '0'
    DEAD_STANDING_TREES = '1'
    FALLEN_OR_BROKEN_TREES = '2'
    DECAYED_STANDING_LIVING_TREES = '3'
    DAMAGES_ON_THE_STEMS = '4'
    FLOWS_OF_RESIN = '5'
    BROKEN_TOP = '61'
    DEAD_LEADER_BRANCH = '62'
    LEADER_CHANGE_BY_LEADER_DAMAGE = '71'
    MULTIPLE_LEADERS = '72'
    BENT_TOP = '73'
    DEFORMED_STEM = '8'
    DEAD_BRANCHES_IN_LIVING_CROWN = '91'
    BROKEN_BRANCHES_IN_LIVING_CROWN = '92'
    DEFORMED_OR_BENT_BRANCHES_IN_LIVING_CROWN = '93'
    ABNORMAL_DYING_BRANCHES_IN_LOWER_CROWN = 'A'
    LOSS_OF_NEEDLES_LEAVES_OR_SHOOTS = 'B'
    LOSS_OF_NEEDLES_LEAVES_OR_SHOOTS_CURRENT_SEASON = 'B1'
    LOSS_OF_OLDER_NEEDLES = 'B2'
    LOSS_OF_NEEDLES_OF_ALL_AGES = 'B3'
    LOSS_OF_LEAVES = 'B4'
    DISCOLORED_NEEDLES_OR_LEAVES = 'C'
    DISCOLORED_NEEDLES_CURRENT_PERIOD = 'C1'
    DISCOLORED_OLDER_NEEDLES = 'C2'
    DISCOLORED_NEEDLES_OF_ALL_AGES = 'C3'
    DISCOLORED_LEAVES = 'C4'
    DEFORMED_NEEDLES_OR_LEAVES = 'D'


class VmiDevelopmentClass(Enum):
    NON_STOCKED_REGENERATION = '1'
    YOUNG_SEEDLING_STAND = '2'
    ADVANCED_SEEDLING_STAND = '3'
    YOUNG_THINNING_STAGE_STAND = '4'
    ADVANCED_THINNING_STAGE_STAND = '5'
    MATURE_STAND = '6'
    SHELTER_TREE_STAND = '7'
    SEED_TREE_STAND = '8'
    UNEVEN_AGED_STAND = '9'


class VmiCuttingMethod(Enum):
    NO_CUTTING = '0'
    TENDING_OF_SEEDLING_STAND = '1'
    FIRST_THINNING = '3'
    OTHER_THINNING = '4'
    OVER_STOREY_THINNING = '5'
    OVER_STOREY_REMOVAL = '6'
    CUTTING_FOR_ARTIFICIAL_REGENERATION = '7'
    CUTTING_FOR_NATURAL_REGENERATION = '8'
    NURSE_CROP_CUTTING = '9'
    SPECIAL_CUTTING = 'A'
    NO_PROPOSED_CUTTING = 'B'

class VmiTimeOfCutting(Enum):
    ONGOING_SEASON = '0'
    PREVIOUS_SEASON = '1'
    TWO_SEASONS_AGO = '2'
    THREE_SEASONS_AGO = '3'
    FOUR_SEASONS_AGO = '4'
    FIVE_SEASONS_AGO = '5'
    SIX_TO_TEN_SEASONS_AGO = '6'
    ELEVEN_TO_THIRTY_SEASONS_AGO = 'A'
    MORE_THAN_THIRTY_YEARS_AGO = 'B'
