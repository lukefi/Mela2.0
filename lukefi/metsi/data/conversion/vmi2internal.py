from typing import Callable, Optional

from lukefi.metsi.data.enums.vmi import (
    VmiCrownClass,
    VmiOrigin,
    VmiArableLandDetail,
    VmiBuildUpLandDetail,
    VmiCuttingMethod,
    VmiDamageType,
    VmiDevelopmentClass,
    VmiFraLandUseClass,
    VmiInlandWaterDetail,
    VmiLandUseCategoryDetail,
    VmiOtherForestryLandDetail,
    VmiPoorlyProductiveForestLandDetail,
    VmiProductiveForestLandDetail,
    VmiSiteType,
    VmiOwnerCategory,
    VmiSoilPeatlandCategory,
    VmiSpecies,
    VmiLandUseCategory,
    VmiTreeCategory,
    VmiDrainageCategory,
    VmiStratumRank,
    VmiTreeStorey,
    VmiTimeOfCutting,
    VmiTreeType,
    VmiUnproductiveLandDetail,
)
from lukefi.metsi.data.enums.internal import (
    CrownClass,
    Origin,
    ArableLandDetail,
    BuildUpLandDetail,
    CuttingMethod,
    DamageType,
    DevelopmentClass,
    FraLandUseClass,
    InlandWaterDetail,
    LandUseCategoryDetail,
    OtherForestryLandDetail,
    PoorlyProductiveForestLandDetail,
    ProductiveForestLandDetail,
    SiteType,
    OwnerCategory,
    SoilPeatlandCategory,
    StratumRank,
    TreeCategory,
    TreeSpecies,
    LandUseCategory,
    DrainageCategory, Storey,
    TreeType,
    UnproductiveLandDetail,
)

_SPECIES_MAP = {
    VmiSpecies.PINE: TreeSpecies.PINE,
    VmiSpecies.SPRUCE: TreeSpecies.SPRUCE,
    VmiSpecies.SILVER_BIRCH: TreeSpecies.SILVER_BIRCH,
    VmiSpecies.DOWNY_BIRCH: TreeSpecies.DOWNY_BIRCH,
    VmiSpecies.ASPEN: TreeSpecies.ASPEN,
    VmiSpecies.GREY_ALDER: TreeSpecies.GREY_ALDER,
    VmiSpecies.COMMON_ALDER: TreeSpecies.COMMON_ALDER,
    VmiSpecies.MOUNTAIN_ASH: TreeSpecies.MOUNTAIN_ASH,
    VmiSpecies.GOAT_WILLOW: TreeSpecies.GOAT_WILLOW,
    VmiSpecies.OTHER_CONIFEROUS: TreeSpecies.OTHER_CONIFEROUS,
    VmiSpecies.SHORE_PINE: TreeSpecies.SHORE_PINE,
    VmiSpecies.KEDAR: TreeSpecies.KEDAR,
    VmiSpecies.OTHER_PINE: TreeSpecies.OTHER_PINE,
    VmiSpecies.LARCH: TreeSpecies.LARCH,
    VmiSpecies.ABIES: TreeSpecies.ABIES,
    VmiSpecies.OTHER_SPRUCE: TreeSpecies.OTHER_SPRUCE,
    VmiSpecies.THUJA: TreeSpecies.THUJA,
    VmiSpecies.JUNIPER: TreeSpecies.JUNIPER,
    VmiSpecies.YEW: TreeSpecies.YEW,
    VmiSpecies.OTHER_DECIDUOUS: TreeSpecies.OTHER_DECIDUOUS,
    VmiSpecies.BAY_WILLOW: TreeSpecies.BAY_WILLOW,
    VmiSpecies.EUROPEAN_WHITE_ELM: TreeSpecies.EUROPEAN_WHITE_ELM,
    VmiSpecies.WYCH_ELM: TreeSpecies.WYCH_ELM,
    VmiSpecies.SMALL_LEAVED_LIME: TreeSpecies.SMALL_LEAVED_LIME,
    VmiSpecies.POPLAR: TreeSpecies.POPLAR,
    VmiSpecies.COMMON_ASH: TreeSpecies.COMMON_ASH,
    VmiSpecies.OAK: TreeSpecies.OAK,
    VmiSpecies.BIRD_CHERRY: TreeSpecies.BIRD_CHERRY,
    VmiSpecies.MAPLE: TreeSpecies.MAPLE,
    VmiSpecies.HAZEL: TreeSpecies.HAZEL,
    VmiSpecies.UNKNOWN: TreeSpecies.UNKNOWN,
    VmiSpecies.TREELESS: TreeSpecies.TREELESS
}

_FRA_LAND_USE_CLASS_MAP = {
    VmiFraLandUseClass.FOREST: FraLandUseClass.FOREST,
    VmiFraLandUseClass.OTHER_WOODED_LAND: FraLandUseClass.OTHER_WOODED_LAND,
    VmiFraLandUseClass.OTHER_LAND: FraLandUseClass.OTHER_LAND,
    VmiFraLandUseClass.OTHER_LAND_WITH_TREE_COVER: FraLandUseClass.OTHER_LAND_WITH_TREE_COVER,
}


_LAND_USE_MAP = {
    VmiLandUseCategory.FOREST: LandUseCategory.FOREST,
    VmiLandUseCategory.SCRUB_LAND: LandUseCategory.SCRUB_LAND,
    VmiLandUseCategory.WASTE_LAND: LandUseCategory.WASTE_LAND,
    VmiLandUseCategory.OTHER_FOREST: LandUseCategory.OTHER_FOREST,
    VmiLandUseCategory.AGRICULTURAL: LandUseCategory.AGRICULTURAL,
    VmiLandUseCategory.BUILT_LAND: LandUseCategory.BUILT_LAND,
    VmiLandUseCategory.ROAD: LandUseCategory.ROAD,
    VmiLandUseCategory.ENERGY_TRANSMISSION_LINE: LandUseCategory.ENERGY_TRANSMISSION_LINE,
    VmiLandUseCategory.FRESHWATER: LandUseCategory.FRESHWATER,
    VmiLandUseCategory.SEA: LandUseCategory.SEA,
    VmiLandUseCategory.OBSOLETE: LandUseCategory.OTHER_FOREST
}

_LAND_USE_DETAIL_MAP: dict[VmiLandUseCategoryDetail, LandUseCategoryDetail] = {
    VmiProductiveForestLandDetail.NORMAL: ProductiveForestLandDetail.NORMAL,
    VmiProductiveForestLandDetail.PASTURE: ProductiveForestLandDetail.PASTURE,
    VmiProductiveForestLandDetail.CONVERTED: ProductiveForestLandDetail.CONVERTED,

    VmiPoorlyProductiveForestLandDetail.NORMAL: PoorlyProductiveForestLandDetail.NORMAL,
    VmiPoorlyProductiveForestLandDetail.PASTURE: PoorlyProductiveForestLandDetail.PASTURE,
    VmiPoorlyProductiveForestLandDetail.CONVERTED: PoorlyProductiveForestLandDetail.CONVERTED,

    VmiUnproductiveLandDetail.VEGETATION_COVER: UnproductiveLandDetail.VEGETATION_COVER,
    VmiUnproductiveLandDetail.NO_VEGETATION_COVER: UnproductiveLandDetail.NO_VEGETATION_COVER,

    VmiOtherForestryLandDetail.OTHER_MINERAL_SOIL: OtherForestryLandDetail.OTHER_MINERAL_SOIL,
    VmiOtherForestryLandDetail.OTHER_ORGANIC_SOIL: OtherForestryLandDetail.OTHER_ORGANIC_SOIL,
    VmiOtherForestryLandDetail.SEED_PRODUCTION_FOREST: OtherForestryLandDetail.SEED_PRODUCTION_FOREST,
    VmiOtherForestryLandDetail.FORESTRY_BUILDINGS: OtherForestryLandDetail.FORESTRY_BUILDINGS,
    VmiOtherForestryLandDetail.FOREST_ROAD: OtherForestryLandDetail.FOREST_ROAD,
    VmiOtherForestryLandDetail.SMALL_SCALE_PEAT_PRODUCTION: OtherForestryLandDetail.SMALL_SCALE_PEAT_PRODUCTION,
    VmiOtherForestryLandDetail.GRAVEL_OR_SAND_PRODUCTION: OtherForestryLandDetail.GRAVEL_OR_SAND_PRODUCTION,

    VmiArableLandDetail.FARMED_FIELDS_OR_FALLOWS: ArableLandDetail.FARMED_FIELDS_OR_FALLOWS,
    VmiArableLandDetail.ABANDONED_FIELD_MINERAL_SOIL: ArableLandDetail.ABANDONED_FIELD_MINERAL_SOIL,
    VmiArableLandDetail.ABANDONED_FIELD_ORGANIC_SOIL: ArableLandDetail.ABANDONED_FIELD_ORGANIC_SOIL,
    VmiArableLandDetail.ABANDONED_FIELD_REFORESTING_MINERAL_SOIL:
        ArableLandDetail.ABANDONED_FIELD_REFORESTING_MINERAL_SOIL,
    VmiArableLandDetail.ABANDONED_FIELD_REFORESTING_ORGANIC_SOIL:
        ArableLandDetail.ABANDONED_FIELD_REFORESTING_ORGANIC_SOIL,
    VmiArableLandDetail.PASTURE_MEADOW: ArableLandDetail.PASTURE_MEADOW,
    VmiArableLandDetail.BIOENERGY_PRODUCTION_WOODY_PLANTS_MINERAL_SOIL:
        ArableLandDetail.BIOENERGY_PRODUCTION_WOODY_PLANTS_MINERAL_SOIL,
    VmiArableLandDetail.BIOENERGY_PRODUCTION_WOODY_PLANTS_ORGANIC_SOIL:
        ArableLandDetail.BIOENERGY_PRODUCTION_WOODY_PLANTS_ORGANIC_SOIL,
    VmiArableLandDetail.BIOENERGY_PRODUCTION_NON_WOODY_PLANTS_MINERAL_SOIL:
        ArableLandDetail.BIOENERGY_PRODUCTION_NON_WOODY_PLANTS_MINERAL_SOIL,
    VmiArableLandDetail.BIOENERGY_PRODUCTION_NON_WOODY_PLANTS_ORGANIC_SOIL:
        ArableLandDetail.BIOENERGY_PRODUCTION_NON_WOODY_PLANTS_ORGANIC_SOIL,
    VmiArableLandDetail.UNPRODUCTIVE_LAND: ArableLandDetail.UNPRODUCTIVE_LAND,
    VmiArableLandDetail.BARNS_AND_OTHER_AGRICULTURAL_BUILDINGS: ArableLandDetail.BARNS_AND_OTHER_AGRICULTURAL_BUILDINGS,
    VmiArableLandDetail.FRUIT_TREE_OR_BERRY_SHRUB_PLANTATION: ArableLandDetail.FRUIT_TREE_OR_BERRY_SHRUB_PLANTATION,

    VmiBuildUpLandDetail.PEAT_PRODUCTION_PREPARATION: BuildUpLandDetail.PEAT_PRODUCTION_PREPARATION,
    VmiBuildUpLandDetail.PEAT_PRODUCTION_OUT_OF_USE: BuildUpLandDetail.PEAT_PRODUCTION_OUT_OF_USE,
    VmiBuildUpLandDetail.PEAT_PRODUCTION_MAINTENANCE: BuildUpLandDetail.PEAT_PRODUCTION_MAINTENANCE,
    VmiBuildUpLandDetail.SURFACE_DRAINAGE_ON_PEAT_PRODUCTION: BuildUpLandDetail.SURFACE_DRAINAGE_ON_PEAT_PRODUCTION,
    VmiBuildUpLandDetail.GREEN_HOUSE_YARD_HOME_GARDEN: BuildUpLandDetail.GREEN_HOUSE_YARD_HOME_GARDEN,
    VmiBuildUpLandDetail.MINING_AREA: BuildUpLandDetail.MINING_AREA,
    VmiBuildUpLandDetail.PEAT_PRODUCTION_ONGOING: BuildUpLandDetail.PEAT_PRODUCTION_ONGOING,
    VmiBuildUpLandDetail.GRAVEL_OR_SAND_PRODUCTION: BuildUpLandDetail.GRAVEL_OR_SAND_PRODUCTION,
    VmiBuildUpLandDetail.OTHER_BUILD_UP_LAND: BuildUpLandDetail.OTHER_BUILD_UP_LAND,

    VmiInlandWaterDetail.NATURAL_WATER_BASIN: InlandWaterDetail.NATURAL_WATER_BASIN,
    VmiInlandWaterDetail.ARTIFICIAL_LAKE_OR_TAMED_RIVER: InlandWaterDetail.ARTIFICIAL_LAKE_OR_TAMED_RIVER,
}


_OWNER_MAP = {
    VmiOwnerCategory.UNKNOWN: OwnerCategory.UNKNOWN,
    VmiOwnerCategory.PRIVATE: OwnerCategory.PRIVATE,
    VmiOwnerCategory.FOREST_INDUSTRY_ENTERPRISE: OwnerCategory.FOREST_INDUSTRY,
    VmiOwnerCategory.OTHER_ENTERPRISE: OwnerCategory.OTHER_ENTERPRISE,
    VmiOwnerCategory.METSAHALLITUS: OwnerCategory.METSAHALLITUS,
    VmiOwnerCategory.OTHER_STATE_AGENCY: OwnerCategory.OTHER_STATE_AGENCY,
    VmiOwnerCategory.FOREST_COOP: OwnerCategory.FOREST_COOP,
    VmiOwnerCategory.MUNICIPALITY: OwnerCategory.MUNICIPALITY,
    VmiOwnerCategory.CONGREGATION: OwnerCategory.CONGREGATION,
    VmiOwnerCategory.OTHER_COMMUNITY: OwnerCategory.OTHER_COMMUNITY,
    VmiOwnerCategory.UNDIVIDED: OwnerCategory.UNDIVIDED
}


_SOIL_PEATLAND_MAP = {
    VmiSoilPeatlandCategory.MINERAL_SOIL: SoilPeatlandCategory.MINERAL_SOIL,
    VmiSoilPeatlandCategory.SPRUCE_MIRE: SoilPeatlandCategory.SPRUCE_MIRE,
    VmiSoilPeatlandCategory.PINE_MIRE: SoilPeatlandCategory.PINE_MIRE,
    VmiSoilPeatlandCategory.TREELESS_MIRE: SoilPeatlandCategory.TREELESS_MIRE,
}


_SITE_TYPE_MAP = {
    VmiSiteType.LEHTO: SiteType.VERY_RICH_SITE,
    VmiSiteType.LEHTOMAINEN_KANGAS: SiteType.RICH_SITE,
    VmiSiteType.TUOREKANGAS: SiteType.DAMP_SITE,
    VmiSiteType.KUIVAHKOKANGAS: SiteType.SUB_DRY_SITE,
    VmiSiteType.KUIVAKANGAS: SiteType.DRY_SITE,
    VmiSiteType.KARUKKOKANGAS: SiteType.BARREN_SITE,
    VmiSiteType.KALLIOMAA_TAI_HIETIKKO: SiteType.ROCKY_OR_SANDY_AREA,
    VmiSiteType.LAKIMETSA_TAI_TUNTURIHAVUMETSA: SiteType.LAKIMETSA_TAI_TUNTURIHAVUMETSA,
    VmiSiteType.TUNTURIKOIVIKKO: SiteType.TUNTURIKOIVIKKO,
    VmiSiteType.AVOTUNTURI: SiteType.OPEN_MOUNTAINS
}


_DRAINAGE_CATEGORY_MAP = {
    VmiDrainageCategory.OJITTAMATON_KANGAS_TAI_SUO: DrainageCategory.UNDRAINED_MINERAL_SOIL_OR_MIRE,
    VmiDrainageCategory.OJITETTU_KANGAS: DrainageCategory.DITCHED_MINERAL_SOIL,
    VmiDrainageCategory.OJIKKO: DrainageCategory.DITCHED_MIRE,
    VmiDrainageCategory.MUUTTUMA: DrainageCategory.TRANSFORMING_MIRE,
    VmiDrainageCategory.TURVEKANGAS: DrainageCategory.TRANSFORMED_MIRE
}

_STRATUM_RANK_MAP = {
    VmiStratumRank.UNPRODUCTIVE_SEEDLINGS: StratumRank.UNGROWABLE_SAPLINGS,
    VmiStratumRank.DOMINANT_STOREY: StratumRank.DOMINANT,
    VmiStratumRank.OVER_STOREY: StratumRank.OVER_1,
    VmiStratumRank.RETENTION_TREE_STOREY: StratumRank.OVER_2,
    VmiStratumRank.NURSE_CROP: StratumRank.OVER_3,
    VmiStratumRank.UNDER_STOREY_CAPABLE_FOR_DEVELOPMENT: StratumRank.UNDER_1,
    VmiStratumRank.UNDER_STOREY_NOT_CAPABLE_FOR_DEVELOPMENT: StratumRank.UNDER_2,
    VmiStratumRank.NON_ESTABLISHED_SEEDLINGS: StratumRank.UNDER_3,
    VmiStratumRank.SEEDLING_STRATUM: StratumRank.UNDER_4,
    VmiStratumRank.DAMAGED_TREE_STRATUM: StratumRank.REMOVAL
}


_TREE_STOREY_MAP = {
    VmiTreeStorey.DOMINANT_MAIN: Storey.DOMINANT,
    VmiTreeStorey.DOMINANT_MIDDLE: Storey.DOMINANT,
    VmiTreeStorey.DOMINANT_LOWER: Storey.DOMINANT,
    VmiTreeStorey.UNDER: Storey.UNDER,
    VmiTreeStorey.OVER_MAIN: Storey.OVER,
    VmiTreeStorey.OVER_OTHER: Storey.OVER,
    VmiTreeStorey.DOMINANT_SPARE_1: Storey.INDETERMINATE,
    VmiTreeStorey.DOMINANT_SPARE_2: Storey.INDETERMINATE,
    VmiTreeStorey.DOMINANT_SPARE_3: Storey.INDETERMINATE,
    VmiTreeStorey.UNDER_SPARE_1: Storey.INDETERMINATE,
    VmiTreeStorey.OVER_SPARE_1: Storey.SPARE,
    VmiTreeStorey.OVER_SPARE_2: Storey.SPARE
}

_ORIGIN_MAP = {
    VmiOrigin.UNKNOWN: Origin.NATURAL,
    VmiOrigin.NATURAL_SEED: Origin.NATURAL,
    VmiOrigin.NATURAL_SPROUT: Origin.NATURAL,
    VmiOrigin.PLANTED: Origin.PLANTED,
    VmiOrigin.SEEDED: Origin.SEEDED
}


_TREE_TYPE_MAP = {
    VmiTreeType.REMEASURED_TALLY_TREE: TreeType.REMEASURED_TALLY_TREE,
    VmiTreeType.NEW_TALLY_TREE_INCREMENT_HEIGHT_GREATER_THAN_1_3_M: TreeType.NEW_TALLY_TREE_INCREMENT_HEIGHT_GREATER_THAN_1_3_M,
    VmiTreeType.NEW_TALLY_TREE_INCREMENT_HEIGHT_LESS_THAN_1_3_M: TreeType.NEW_TALLY_TREE_INCREMENT_HEIGHT_LESS_THAN_1_3_M,
    VmiTreeType.NEW_TALLY_TREE_OTHER_THAN_INCREMENT: TreeType.NEW_TALLY_TREE_OTHER_THAN_INCREMENT,
    VmiTreeType.OLD_TALLY_TREE_STUMP_STEM_REMOVED: TreeType.OLD_TALLY_TREE_STUMP_STEM_REMOVED,
    VmiTreeType.OLD_TALLY_TREE_STUMP_STEM_NOT_REMOVED: TreeType.OLD_TALLY_TREE_STUMP_STEM_NOT_REMOVED,
    VmiTreeType.OLD_TALLY_TREE_MEASURED_PREVIOUSLY_BY_MISTAKE: TreeType.OLD_TALLY_TREE_MEASURED_PREVIOUSLY_BY_MISTAKE,
    VmiTreeType.OLD_TALLY_TREE_MEASURED_PREVIOUSLY_BY_MISTAKE_NO_LONGER_TALLY: TreeType.OLD_TALLY_TREE_MEASURED_PREVIOUSLY_BY_MISTAKE_NO_LONGER_TALLY,
    VmiTreeType.OLD_TALLY_TREE_LAND_USE_CLASS_CHANGED_NO_LONGER_EXISTS: TreeType.OLD_TALLY_TREE_LAND_USE_CLASS_CHANGED_NO_LONGER_EXISTS,
    VmiTreeType.OLD_TALLY_TREE_LAND_USE_CLASS_CHANGED_STILL_EXISTS: TreeType.OLD_TALLY_TREE_LAND_USE_CLASS_CHANGED_STILL_EXISTS,
    VmiTreeType.OLD_TALLY_TREE_NOT_FOUND: TreeType.OLD_TALLY_TREE_NOT_FOUND,
    VmiTreeType.OLD_TALLY_TREE_NOW_OUTSIDE_PLOT: TreeType.OLD_TALLY_TREE_NOW_OUTSIDE_PLOT,
    VmiTreeType.OLD_TALLY_TREE_NOW_OUT_OF_PLOT_AREA: TreeType.OLD_TALLY_TREE_NOW_OUT_OF_PLOT_AREA,
    VmiTreeType.OLD_TALLY_TREE_NOW_OUT_OF_PLOT_AREA_DUE_TO_DIAMETER_OR_DISTANCE: TreeType.OLD_TALLY_TREE_NOW_OUT_OF_PLOT_AREA_DUE_TO_DIAMETER_OR_DISTANCE,
    VmiTreeType.OLD_CHECKED_TALLY_TREE: TreeType.OLD_CHECKED_TALLY_TREE}

_TREE_CATEGORY_MAP = {
    VmiTreeCategory.SMALL_TREE: TreeCategory.SMALL_TREE,
    VmiTreeCategory.WASTE_TREE: TreeCategory.WASTE_TREE,
    VmiTreeCategory.PULP_WOOD_TREE: TreeCategory.PULP_WOOD_TREE,
    VmiTreeCategory.SAW_LOG_TREE: TreeCategory.SAW_LOG_TREE,
    VmiTreeCategory.USABLE_STANDING_DEAD_TREE: TreeCategory.USABLE_STANDING_DEAD_TREE,
    VmiTreeCategory.USABLE_FALLEN_DEAD_TREE: TreeCategory.USABLE_FALLEN_DEAD_TREE,
    VmiTreeCategory.UNUSABLE_DEAD_TREE: TreeCategory.UNUSABLE_DEAD_TREE,
    VmiTreeCategory.STUMP_ALIVE_WHEN_FELLING: TreeCategory.STUMP_ALIVE_WHEN_FELLING,
    VmiTreeCategory.STUMP_DEAD_STANDING_WHEN_FELLING: TreeCategory.STUMP_DEAD_STANDING_WHEN_FELLING,
    VmiTreeCategory.STUMP_DEAD_FALLEN_WHEN_FELLING: TreeCategory.STUMP_DEAD_FALLEN_WHEN_FELLING,
}

_DAMAGE_TYPE_MAP = {
    VmiDamageType.NO_DAMAGE: DamageType.NO_DAMAGE,
    VmiDamageType.DEAD_STANDING_TREES: DamageType.DEAD_STANDING_TREES,
    VmiDamageType.FALLEN_OR_BROKEN_TREES: DamageType.FALLEN_OR_BROKEN_TREES,
    VmiDamageType.DECAYED_STANDING_LIVING_TREES: DamageType.DECAYED_STANDING_LIVING_TREES,
    VmiDamageType.DAMAGES_ON_THE_STEMS: DamageType.DAMAGES_ON_THE_STEMS,
    VmiDamageType.FLOWS_OF_RESIN: DamageType.FLOWS_OF_RESIN,
    VmiDamageType.BROKEN_TOP: DamageType.BROKEN_TOP,
    VmiDamageType.DEAD_LEADER_BRANCH: DamageType.DEAD_LEADER_BRANCH,
    VmiDamageType.LEADER_CHANGE_BY_LEADER_DAMAGE: DamageType.LEADER_CHANGE_BY_LEADER_DAMAGE,
    VmiDamageType.MULTIPLE_LEADERS: DamageType.MULTIPLE_LEADERS,
    VmiDamageType.BENT_TOP: DamageType.BENT_TOP,
    VmiDamageType.DEFORMED_STEM: DamageType.DEFORMED_STEM,
    VmiDamageType.DEAD_BRANCHES_IN_LIVING_CROWN: DamageType.DEAD_BRANCHES_IN_LIVING_CROWN,
    VmiDamageType.BROKEN_BRANCHES_IN_LIVING_CROWN: DamageType.BROKEN_BRANCHES_IN_LIVING_CROWN,
    VmiDamageType.DEFORMED_OR_BENT_BRANCHES_IN_LIVING_CROWN: DamageType.DEFORMED_OR_BENT_BRANCHES_IN_LIVING_CROWN,
    VmiDamageType.ABNORMAL_DYING_BRANCHES_IN_LOWER_CROWN: DamageType.ABNORMAL_DYING_BRANCHES_IN_LOWER_CROWN,
    VmiDamageType.LOSS_OF_NEEDLES_LEAVES_OR_SHOOTS: DamageType.LOSS_OF_NEEDLES_LEAVES_OR_SHOOTS,
    VmiDamageType.LOSS_OF_NEEDLES_LEAVES_OR_SHOOTS_CURRENT_SEASON:
        DamageType.LOSS_OF_NEEDLES_LEAVES_OR_SHOOTS_CURRENT_SEASON,
    VmiDamageType.LOSS_OF_OLDER_NEEDLES: DamageType.LOSS_OF_OLDER_NEEDLES,
    VmiDamageType.LOSS_OF_NEEDLES_OF_ALL_AGES: DamageType.LOSS_OF_NEEDLES_OF_ALL_AGES,
    VmiDamageType.LOSS_OF_LEAVES: DamageType.LOSS_OF_LEAVES,
    VmiDamageType.DISCOLORED_NEEDLES_OR_LEAVES: DamageType.DISCOLORED_NEEDLES_OR_LEAVES,
    VmiDamageType.DISCOLORED_NEEDLES_CURRENT_PERIOD: DamageType.DISCOLORED_NEEDLES_CURRENT_PERIOD,
    VmiDamageType.DISCOLORED_OLDER_NEEDLES: DamageType.DISCOLORED_OLDER_NEEDLES,
    VmiDamageType.DISCOLORED_NEEDLES_OF_ALL_AGES: DamageType.DISCOLORED_NEEDLES_OF_ALL_AGES,
    VmiDamageType.DISCOLORED_LEAVES: DamageType.DISCOLORED_LEAVES,
    VmiDamageType.DEFORMED_NEEDLES_OR_LEAVES: DamageType.DEFORMED_NEEDLES_OR_LEAVES,
}


_DEVELOPMENT_CLASS_MAP = {
    VmiDevelopmentClass.NON_STOCKED_REGENERATION: DevelopmentClass.NON_STOCKED_REGENERATION,
    VmiDevelopmentClass.YOUNG_SEEDLING_STAND: DevelopmentClass.YOUNG_SEEDLING_STAND,
    VmiDevelopmentClass.ADVANCED_SEEDLING_STAND: DevelopmentClass.ADVANCED_SEEDLING_STAND,
    VmiDevelopmentClass.YOUNG_THINNING_STAGE_STAND: DevelopmentClass.YOUNG_THINNING_STAGE_STAND,
    VmiDevelopmentClass.ADVANCED_THINNING_STAGE_STAND: DevelopmentClass.ADVANCED_THINNING_STAGE_STAND,
    VmiDevelopmentClass.MATURE_STAND: DevelopmentClass.MATURE_STAND,
    VmiDevelopmentClass.SHELTER_TREE_STAND: DevelopmentClass.SHELTER_TREE_STAND,
    VmiDevelopmentClass.SEED_TREE_STAND: DevelopmentClass.SEED_TREE_STAND,
    VmiDevelopmentClass.UNEVEN_AGED_STAND: DevelopmentClass.UNEVEN_AGED_STAND,
}


_CUTTING_METHOD_MAP = {
    VmiCuttingMethod.NO_CUTTING: CuttingMethod.NO_CUTTING,
    VmiCuttingMethod.OTHER_THINNING: CuttingMethod.THINNING,
    VmiCuttingMethod.CUTTING_FOR_ARTIFICIAL_REGENERATION: CuttingMethod.CLEARCUTTING,
    VmiCuttingMethod.FIRST_THINNING: CuttingMethod.FIRST_THINNING,
    VmiCuttingMethod.OVER_STOREY_REMOVAL: CuttingMethod.OVER_STORY_REMOVAL,
    VmiCuttingMethod.CUTTING_FOR_NATURAL_REGENERATION: CuttingMethod.SEED_TREE_CUTTING,
    VmiCuttingMethod.NURSE_CROP_CUTTING: CuttingMethod.SHELTERWOOD_CUTTING
}


_CROWN_CLASS_MAP = {
    VmiCrownClass.CROWNLESS: CrownClass.CROWNLESS,
    VmiCrownClass.DOMINANT_TREE_IN_DOMINANT_TREE_STOREY: CrownClass.DOMINANT_TREE_IN_DOMINANT_TREE_STOREY,
    VmiCrownClass.INTERMEDIATE_TREE_IN_DOMINANT_TREE_STOREY: CrownClass.INTERMEDIATE_TREE_IN_DOMINANT_TREE_STOREY,
    VmiCrownClass.SUPPRESSED_TREE_IN_DOMINANT_TREE_STOREY: CrownClass.SUPPRESSED_TREE_IN_DOMINANT_TREE_STOREY,
    VmiCrownClass.UNDER_STOREY_TREE: CrownClass.UNDER_STOREY_TREE,
    VmiCrownClass.DOMINANT_TREE_IN_OVER_STOREY: CrownClass.DOMINANT_TREE_IN_OVER_STOREY,
    VmiCrownClass.INTERMEDIATE_OR_SUPPRESSED_TREE_IN_OVER_STOREY: CrownClass.INTERMEDIATE_OR_SUPPRESSED_TREE_IN_OVER_STOREY,
    VmiCrownClass.RETENTION_DOMINANT_TREE_IN_DOMINANT_TREE_STOREY: CrownClass.RETENTION_DOMINANT_TREE_IN_DOMINANT_TREE_STOREY,
    VmiCrownClass.RETENTION_INTERMEDIATE_TREE_IN_DOMINANT_TREE_STOREY: CrownClass.RETENTION_INTERMEDIATE_TREE_IN_DOMINANT_TREE_STOREY,
    VmiCrownClass.RETENTION_SUPPRESSED_TREE_IN_DOMINANT_TREE_STOREY: CrownClass.RETENTION_SUPPRESSED_TREE_IN_DOMINANT_TREE_STOREY,
    VmiCrownClass.RETENTION_UNDER_STOREY_TREE: CrownClass.RETENTION_UNDER_STOREY_TREE,
    VmiCrownClass.RETENTION_DOMINANT_TREE_IN_OVER_STOREY: CrownClass.RETENTION_DOMINANT_TREE_IN_OVER_STOREY,
    VmiCrownClass.RETENTION_INTERMEDIATE_OR_SUPPRESSED_TREE_IN_OVER_STOREY: CrownClass.RETENTION_INTERMEDIATE_OR_SUPPRESSED_TREE_IN_OVER_STOREY
}


def check_empty_vmi[T](func: Callable[[str], T]) -> Callable[[str], Optional[T]]:
    def inner(code: str):
        if code in ('', ' ', '.'):
            return None
        return func(code)
    return inner


@check_empty_vmi
def convert_drainage_category(code: str) -> DrainageCategory:
    value = VmiDrainageCategory(code)
    return _DRAINAGE_CATEGORY_MAP[value]


@check_empty_vmi
def convert_site_type_category(code: str) -> SiteType:
    value = VmiSiteType(code)
    return _SITE_TYPE_MAP[value]


@check_empty_vmi
def convert_soil_peatland_category(code: str) -> SoilPeatlandCategory:
    vmi_category = VmiSoilPeatlandCategory(code)
    return _SOIL_PEATLAND_MAP[vmi_category]


@check_empty_vmi
def convert_fra_land_use_class(fra_code: str) -> FraLandUseClass:
    vmi_fra = VmiFraLandUseClass(fra_code)
    return _FRA_LAND_USE_CLASS_MAP[vmi_fra]


def convert_land_use_category(lu_code: str) -> LandUseCategory:
    """sanitization of lu_code is the responsibility of the caller,
    meaning that this conversion will fail e.g. if the parameter is a lower-case letter."""
    vmi_category = VmiLandUseCategory(lu_code)
    return _LAND_USE_MAP[vmi_category]


def convert_land_use_category_detail(lu_cat: LandUseCategory, lud_code: str) -> Optional[LandUseCategoryDetail]:
    vmi_cat_det: VmiLandUseCategoryDetail
    if lu_cat == LandUseCategory.FOREST:
        vmi_cat_det = VmiProductiveForestLandDetail(lud_code)
    elif lu_cat == LandUseCategory.SCRUB_LAND:
        vmi_cat_det = VmiPoorlyProductiveForestLandDetail(lud_code)
    elif lu_cat == LandUseCategory.WASTE_LAND:
        vmi_cat_det = VmiUnproductiveLandDetail(lud_code)
    elif lu_cat == LandUseCategory.OTHER_FOREST:
        vmi_cat_det = VmiOtherForestryLandDetail(lud_code)
    elif lu_cat == LandUseCategory.AGRICULTURAL:
        vmi_cat_det = VmiArableLandDetail(lud_code)
    elif lu_cat == LandUseCategory.BUILT_LAND:
        vmi_cat_det = VmiBuildUpLandDetail(lud_code)
    elif lu_cat == LandUseCategory.FRESHWATER:
        vmi_cat_det = VmiInlandWaterDetail(lud_code)
    else:
        return None

    return _LAND_USE_DETAIL_MAP[vmi_cat_det]


def convert_species(species_code: str) -> TreeSpecies:
    """Converts VMI species code to internal TreeSpecies code"""
    value = species_code.strip()
    vmi_species = VmiSpecies(value)
    return _SPECIES_MAP[vmi_species]


def convert_owner(owner_code: str) -> OwnerCategory:
    vmi_owner = VmiOwnerCategory(owner_code)
    return _OWNER_MAP[vmi_owner]


@check_empty_vmi
def convert_stratum_rank(rank_code: str) -> StratumRank:
    vmi_rank = VmiStratumRank(rank_code)
    return _STRATUM_RANK_MAP[vmi_rank]


@check_empty_vmi
def convert_tree_storey(storey_code: str) -> Storey:
    vmi_storey = VmiTreeStorey(storey_code)
    return _TREE_STOREY_MAP[vmi_storey]


@check_empty_vmi
def convert_origin(origin_code: str) -> Origin:
    vmi_origin = VmiOrigin(origin_code)
    return _ORIGIN_MAP[vmi_origin]


@check_empty_vmi
def convert_tree_type(type_code: str) -> TreeType:
    vmi_type = VmiTreeType(type_code)
    return _TREE_TYPE_MAP[vmi_type]


@check_empty_vmi
def convert_tree_category(cat_code: str) -> TreeCategory:
    vmi_cat = VmiTreeCategory(cat_code)
    return _TREE_CATEGORY_MAP[vmi_cat]


@check_empty_vmi
def convert_damage_type(dam_code: str) -> DamageType:
    vmi_dam = VmiDamageType(dam_code)
    return _DAMAGE_TYPE_MAP[vmi_dam]


def convert_development_class(dev_code: str) -> DevelopmentClass:
    if dev_code in ('', ' ', '.'):
        return DevelopmentClass.UNKNOWN
    vmi_dev = VmiDevelopmentClass(dev_code)
    return _DEVELOPMENT_CLASS_MAP[vmi_dev]


def _convert_cutting_method(cut_code: str, cutting_year: Optional[int]) -> CuttingMethod:
    if cut_code in ('', ' ', '.'):
        return CuttingMethod.NO_CUTTING
    if cutting_year is not None and cutting_year > 0:
        vmi_cut = VmiCuttingMethod(cut_code)
        return _CUTTING_METHOD_MAP.get(vmi_cut, CuttingMethod.NO_CUTTING)
    return CuttingMethod.NO_CUTTING


def _determine_forest_maintenance_year(cutting_time_src: str, year: int) -> Optional[int]:
    """Determine the year of last operation from given VMI source classes and the year of data set."""
    if cutting_time_src in ('', ' ', '.'):
        return None
    vmi_cutting_time = VmiTimeOfCutting(cutting_time_src)
    if vmi_cutting_time in (
            VmiTimeOfCutting.ONGOING_SEASON,
            VmiTimeOfCutting.PREVIOUS_SEASON,
            VmiTimeOfCutting.TWO_SEASONS_AGO,
            VmiTimeOfCutting.THREE_SEASONS_AGO,
            VmiTimeOfCutting.FOUR_SEASONS_AGO,
            VmiTimeOfCutting.FIVE_SEASONS_AGO):
        return year - int(cutting_time_src)
    if vmi_cutting_time == VmiTimeOfCutting.SIX_TO_TEN_SEASONS_AGO:
        return year - 7
    if vmi_cutting_time == VmiTimeOfCutting.ELEVEN_TO_THIRTY_SEASONS_AGO:
        return year - 20
    if vmi_cutting_time == VmiTimeOfCutting.MORE_THAN_THIRTY_YEARS_AGO:
        return year - 40
    return None


def convert_forest_maintenance_details(cutting_type_class_src: str,
                                       cutting_time_src: str,
                                       year: int) -> tuple[Optional[int], Optional[int], Optional[CuttingMethod]]:
    """
    Return a triplet of (young_stand_tending_year, cutting_year, cutting_method). VMI source data is exclusive
    between cutting and tending, i.e. the codes are overloaded into the same year class variable. RST target format
    allows separate value for both tending and cutting years, but this is impossible in source data.
    """
    operation_year = _determine_forest_maintenance_year(cutting_time_src, year)
    method = _convert_cutting_method(cutting_type_class_src, operation_year)

    if cutting_type_class_src in ('1', '2'):
        return operation_year, None, None
    if method == 0:
        # This case is necessary. Operations over 10 years old are listed as type 0, or no operation in VMI data.
        # The actual year is still recorded, but we don't seem to want it in RST target. This is based on original
        # implementation of this application.
        return None, None, None
    return None, operation_year, method


@check_empty_vmi
def convert_crown_class(crown_str: str) -> CrownClass:
    vmi_crown = VmiCrownClass(crown_str)
    return _CROWN_CLASS_MAP[vmi_crown]
