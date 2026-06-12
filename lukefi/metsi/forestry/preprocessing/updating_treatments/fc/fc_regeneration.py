from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.enums.internal import (
    TreeSpecies,
    SiteType,
    SoilPeatlandCategory,
    Origin
)
from lukefi.metsi.domain.forestry_treatments.regeneration import regeneration_fn
from lukefi.metsi.sim.treatment import PredeterminedTreatment


def _regeneration_species_fn(stand: ForestStand) -> TreeSpecies:
    if stand.site_type_category in (SiteType.VERY_RICH_SITE,
                                    SiteType.RICH_SITE,
                                    ):
        return TreeSpecies.SPRUCE
    if stand.site_type_category == SiteType.DAMP_SITE:
        if stand.soil_peatland_category == SoilPeatlandCategory.MINERAL_SOIL:
            return TreeSpecies.SILVER_BIRCH
        else:
            return TreeSpecies.DOWNY_BIRCH
    return TreeSpecies.PINE


def _seeded_stems_fn(stand: ForestStand) -> int:
    spe = _regeneration_species_fn(stand)
    if spe == TreeSpecies.SPRUCE:
        return 4000
    if spe in (TreeSpecies.SILVER_BIRCH, TreeSpecies.DOWNY_BIRCH):
        return 5000
    return 4500


seeding = PredeterminedTreatment(
    name="fc_seeding",
    treatment_fn=regeneration_fn,
    static_parameters={
        "origin": Origin.SEEDED,
        "height": 0,
        "biological_age": 0,
        "ntrees": 1,
        "type": "artificial"
    },
    dynamic_parameters={
        "species": _regeneration_species_fn,
        "stems_per_ha": _seeded_stems_fn,
    }
)

seedingPine = PredeterminedTreatment(
    name="fc_seedingPine",
    treatment_fn=regeneration_fn,
    static_parameters={
        "origin": Origin.SEEDED,
        "height": 0,
        "biological_age": 0,
        "ntrees": 1,
        "species": TreeSpecies.PINE,
        "stems_per_ha": 4500,
        "type": "artificial"
    },
)

seedingSpruce = PredeterminedTreatment(
    name="fc_seedingPine",
    treatment_fn=regeneration_fn,
    static_parameters={
        "origin": Origin.SEEDED,
        "height": 0,
        "biological_age": 0,
        "ntrees": 1,
        "species": TreeSpecies.SPRUCE,
        "stems_per_ha": 4000,
        "type": "artificial"
    },
)

seedingSilverBirch = PredeterminedTreatment(
    name="fc_seedingPine",
    treatment_fn=regeneration_fn,
    static_parameters={
        "origin": Origin.SEEDED,
        "height": 0,
        "biological_age": 0,
        "ntrees": 1,
        "species": TreeSpecies.SILVER_BIRCH,
        "stems_per_ha": 5000,
        "type": "artificial"
    },
)

seedingDownyBirch = PredeterminedTreatment(
    name="fc_seedingPine",
    treatment_fn=regeneration_fn,
    static_parameters={
        "origin": Origin.SEEDED,
        "height": 0,
        "biological_age": 0,
        "ntrees": 1,
        "species": TreeSpecies.DOWNY_BIRCH,
        "stems_per_ha": 5000,
        "type": "artificial"
    },
)


def _planted_stems_fn(stand: ForestStand) -> int:
    spe = _regeneration_species_fn(stand)
    if spe == TreeSpecies.SPRUCE:
        return 2000
    if spe in (TreeSpecies.SILVER_BIRCH, TreeSpecies.DOWNY_BIRCH):
        return 1600
    return 2400


planting = PredeterminedTreatment(
    name="fc_planting",
    treatment_fn=regeneration_fn,
    static_parameters={
        "origin": Origin.PLANTED,
        "height": 0,
        "biological_age": 0,
        "ntrees": 10,
        "type": "artificial"
    },
    dynamic_parameters={
        "species": _regeneration_species_fn,
        "stems_per_ha": _planted_stems_fn,
    }
)

plantingPine = PredeterminedTreatment(
    name="fc_plantingPine",
    treatment_fn=regeneration_fn,
    static_parameters={
        "origin": Origin.PLANTED,
        "height": 0,
        "biological_age": 0,
        "ntrees": 10,
        "species": TreeSpecies.PINE,
        "stems_per_ha": 4500,
        "type": "artificial"
    },
)

plantingSpruce = PredeterminedTreatment(
    name="fc_plantingSpruce",
    treatment_fn=regeneration_fn,
    static_parameters={
        "origin": Origin.PLANTED,
        "height": 0,
        "biological_age": 0,
        "ntrees": 10,
        "species": TreeSpecies.SPRUCE,
        "stems_per_ha": 4000,
        "type": "artificial"
    },
)

plantingSilverBirch = PredeterminedTreatment(
    name="fc_plantingSilverBirch",
    treatment_fn=regeneration_fn,
    static_parameters={
        "origin": Origin.PLANTED,
        "height": 0,
        "biological_age": 0,
        "ntrees": 10,
        "species": TreeSpecies.SILVER_BIRCH,
        "stems_per_ha": 1600,
        "type": "artificial"
    },
)

plantingDownyBirch = PredeterminedTreatment(
    name="fc_plantingDownyBirch",
    treatment_fn=regeneration_fn,
    static_parameters={
        "origin": Origin.PLANTED,
        "height": 0,
        "biological_age": 0,
        "ntrees": 1,
        "species": TreeSpecies.DOWNY_BIRCH,
        "stems_per_ha": 1600,
        "type": "artificial"
    },
)
