from typing import override
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.conversion import vmi2internal
from lukefi.metsi.data.enums.vmi import VmiIteration
from lukefi.metsi.data.formats import util, vmi_util
from lukefi.metsi.data.formats.declarative_conversion import Conversion
from lukefi.metsi.data.formats.forest_builder_base import RowKind, VMIBuilder
from lukefi.metsi.data.formats.vmi_const import VMI13_STAND_INDICES, VMI13_STRATUM_INDICES, VMI13_TREE_INDICES
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.domain.forestry_types import StandList


class VMI13Builder(VMIBuilder):

    def __init__(self, builder_flags: dict[str, bool],
                 conversions: dict[str, Conversion], data_rows: list[str]) -> None:
        super().__init__(builder_flags, conversions)
        for row in data_rows:
            kind = self._classify_row(row)
            split_row = row.split()

            if kind == RowKind.STAND:
                self.stand_rows.append({key: split_row[index] for key, index in VMI13_STAND_INDICES.items()})

            elif kind == RowKind.STRATUM and self.builder_flags.get("strata", False):
                stratum_row = {key: split_row[index] for key, index in VMI13_STRATUM_INDICES.items()}
                stand_identifier = vmi_util.generate_stand_identifier(stratum_row)
                self.stratum_rows.setdefault(stand_identifier, []).append(stratum_row)

            elif kind == RowKind.TREE and self.builder_flags.get("measured_trees", False):
                tree_row = {key: split_row[index] for key, index in VMI13_TREE_INDICES.items()}
                stand_identifier = vmi_util.generate_stand_identifier(tree_row)
                self.tree_rows.setdefault(stand_identifier, []).append(tree_row)

    @staticmethod
    def _classify_row(row: str) -> RowKind:
        return RowKind(row[0])

    @override
    def build(self) -> StandList:
        result: dict[str, ForestStand] = {}

        for i, row in enumerate(self.stand_rows):
            try:
                stand = self._convert_stand_entry(row, i + 1)
                result[stand.identifier] = stand
            except Exception as e:
                raise MetsiException(f"Parsing stand row {row} failed: {e}") from e

        if self.builder_flags.get('strata', False):
            for stand_identifier, stand_stratum_rows in self.stratum_rows.items():
                strata = TreeStrata(len(stand_stratum_rows))
                for j, stand_stratum_row in enumerate(stand_stratum_rows):
                    try:
                        self._convert_stratum_entry(strata, stand_stratum_row, j)
                    except Exception as e:
                        raise MetsiException(f"Parsing stratum row {stand_stratum_row} failed: {e}") from e

                result[stand_identifier].tree_strata = strata

        if self.builder_flags.get('measured_trees', False):
            for stand_identifier, stand_tree_rows in self.tree_rows.items():
                stand_ = result.get(stand_identifier)
                if stand_ is None:
                    continue

                trees = ReferenceTrees(len(stand_tree_rows))
                for j, stand_tree_row in enumerate(stand_tree_rows):
                    try:
                        self._convert_tree_entry(trees, stand_tree_row, j, stand_.forestry_centre_id)
                    except Exception as e:
                        raise MetsiException(f"Parsing tree row {stand_tree_row} failed: {e}") from e

                result[stand_identifier].reference_trees = trees

        return list(result.values())

    @staticmethod
    def _convert_stand_entry(row: dict[str, str], stand_id: int | None = None) -> ForestStand:
        result = ForestStand()

        result.identifier = vmi_util.generate_stand_identifier(row)
        result.stand_id = stand_id

        result.degree_days = vmi_util.transform_vmi_degree_days(row["degree_days"])
        result.owner_category = vmi2internal.convert_owner(row["owner_group"])
        result.fra_category = vmi2internal.convert_fra_land_use_class(row["fra_class"])
        result.land_use_category = vmi2internal.convert_land_use_category(row["land_category"])
        result.site_type_category = vmi2internal.convert_site_type_category(row["kasvupaikkatunnus"])
        result.soil_peatland_category = vmi2internal.convert_soil_peatland_category(row["paatyyppi"])
        result.tax_class_reduction = vmi_util.determine_tax_class_reduction(row["tax_class_reduction"])
        result.tax_class = vmi_util.determine_tax_class(row["tax_class"])
        result.drainage_category = vmi2internal.convert_drainage_category(row["ojitus_tilanne"])
        result.forestry_centre_id = vmi_util.parse_forestry_centre(row["forestry_centre"])
        result.municipality_id = util.parse_int(vmi_util.vmi_codevalue(row["municipality"]))
        result.auxiliary_stand = row["stand_number"] != '1'

        result.year = vmi_util.parse_vmi13_date(row["date"]).year
        result.start_year = result.year

        area_ha = vmi_util.determine_vmi13_area_ha(
            int(row["county"]),
            int(row["lohkomuoto"]),
            util.get_or_default(
                util.parse_int(row["lohkotarkenne"]),
                0
            )
        )

        result.area_weight_factors = vmi_util.determine_area_factors(
            row["osuus4m"],
            row["osuus9m"]
        )
        result.set_area(area_ha)

        lat = util.get_or_default(util.parse_type(row["lat_measured"], float), 0.0)
        lon = util.get_or_default(util.parse_type(row["lon_measured"], float), 0.0)
        if not lat:
            lat = util.get_or_default(util.parse_type(row["lat"], float), 0.0)
        if not lon:
            lon = util.get_or_default(util.parse_type(row["lon"], float), 0.0)

        height = vmi_util.transform_vmi13_height_above_sea_level(row["height_above_sea_level"])
        result.set_geo_location(lat, lon, height)

        result.drainage_year = vmi_util.determine_drainage_year(row["ojitus_aika"], result.year)
        result.fertilization_year = None
        result.soil_surface_preparation_year = vmi_util.determine_soil_surface_preparation_year(
            row["maanmuokkaus_aika"],
            result.year
        )
        result.regeneration_area_cleaning_year = vmi_util.determine_clearing_of_reform_sector_year(
            row["muu_toimenpide"],
            row["muu_toimenpide_aika"],
            result.year
        )
        result.artificial_regeneration_year = vmi_util.determine_artificial_regeneration_year(
            row["viljely"],
            row["viljely_aika"],
            result.year
        )
        maintenance_details = vmi2internal.convert_forest_maintenance_details(
            row["hakkuu_tapa"],
            row["hakkuu_aika"],
            result.year
        )
        result.young_stand_tending_year = maintenance_details[0]
        result.cutting_year = maintenance_details[1]
        result.method_of_last_cutting = maintenance_details[2]
        result.ds_main_tree_species_biological_age = vmi_util.determine_vmi13_dominant_storey_age(
            row["vallitsevanjaksonika"]
        )

        result.peatland_type = vmi2internal.convert_peatland_forest_type(row["suotyy"])
        result.drained_peatland_type = vmi2internal.convert_drained_peatland_forest_type(row["tkgtyy"])
        result.under_storey = bool(util.parse_type(row["alikehl"], int))
        result.over_storey = bool(util.parse_type(row["ylikehl"], int))

        result.development_class = vmi2internal.convert_development_class(row["kehitysluokka"])
        result.main_tree_species_dominant_storey = vmi_util.determine_main_tree_species_dominant_storey(
            row["main_tree_species_dominant_storey"],
            result.site_type_category
        )
        result.basal_area = util.parse_type(row["pohjapintaala"], float)
        result.region = util.parse_int(row["county"])

        result.municipality_id = vmi_util.determine_municipality(
            row["municipality"],
            row["kitukunta"]
        )

        if result.land_use_category and result.region is not None and result.owner_category is not None:
            is_ahvenanmaa = result.region == 21
            result.forest_management_category = vmi_util.determine_forest_management_category(
                result.land_use_category,
                is_ahvenanmaa,
                result.owner_category,
                row["puuntuotannon_rajoitus"],
                row["puuntuotannon_rajoitus_tarkenne"],
                row["muut_arvot"],
                row["suojametsakoodi"],
                row["ahvenanmaan_markkinahakkuualue"],
                row["koealan_kasittelyluokka"]
            )
        else:
            result.forest_management_category = 1

        return result

    @staticmethod
    def _convert_stratum_entry(strata: TreeStrata, row: dict[str, str], i: int):
        identifier = vmi_util.generate_stratum_identifier(row)

        species = vmi2internal.convert_species(row["species"])
        origin = vmi2internal.convert_origin(row["origin"])

        stems_per_ha = util.get_or_default(util.parse_type(row["stems_per_ha"], float), 0.0)
        sapling_stems_per_ha = util.get_or_default(util.parse_type(row["sapling_stems_per_ha"], float), 0.0)

        mean_diameter = util.parse_type(row["avg_diameter"], float)
        mean_height = vmi_util.determine_stratum_tree_height(row["avg_height"])

        biological_age, breast_height_age = vmi_util.determine_stratum_age_values(
            row["biological_age"],
            row["d13_age"],
            mean_height
        )

        basal_area = util.parse_type(row["basal_area"], float)
        stratum_number = util.parse_int(row["stratum_number"])
        storey = vmi_util.determine_storey_for_stratum(row["stratum_rank"])
        stratum_rank = vmi2internal.convert_stratum_rank(row["stratum_rank"])

        strata.identifier[i] = identifier
        strata.species[i] = species
        strata.mean_diameter[i] = mean_diameter
        strata.mean_height[i] = mean_height
        strata.breast_height_age[i] = breast_height_age
        strata.biological_age[i] = biological_age
        strata.stems_per_ha[i] = stems_per_ha
        strata.basal_area[i] = basal_area
        if origin is not None:
            strata.origin[i] = origin
        if stratum_number is not None:
            strata.stratum_number[i] = stratum_number
        if storey is not None:
            strata.storey[i] = storey
        strata.sapling_stems_per_ha[i] = sapling_stems_per_ha
        if stratum_rank is not None:
            strata.stratum_rank[i] = stratum_rank

    @staticmethod
    def _convert_tree_entry(trees: ReferenceTrees,
                            row: dict[str, str],
                            i: int,
                            forestry_centre_id: int | None,
                            ahvkeilaus: str | None = None,
                            height_conversion_factor: float = 100.0,
                            measured_height_conversion_factor: float = 10.0):
        identifier = vmi_util.generate_tree_identifier(row)
        tree_number = util.parse_type(row["tree_number"], int)
        species = vmi2internal.convert_species(row["species"])
        tree_category = vmi2internal.convert_tree_category(row["tree_category"])

        breast_height_diameter = vmi_util.transform_tree_diameter(row["diameter"])
        breast_height_age, biological_age = vmi_util.determine_tree_age_values(
            row["d13_age"],
            row["age_increase"],
            row["total_age"],
        )

        height = vmi_util.determine_tree_height(
            row["height"],
            conversion_factor=height_conversion_factor,
        )
        measured_height = vmi_util.determine_tree_height(
            row["measured_height"],
            conversion_factor=measured_height_conversion_factor,
        )

        stems_per_ha = vmi_util.determine_stems_per_ha(breast_height_diameter,
                                                       vmi_version=VmiIteration.VMI13,
                                                       forestry_centre_id=forestry_centre_id,
                                                       ahvkeilaus=ahvkeilaus)
        origin = vmi2internal.convert_origin(row["origin"])

        management_category = vmi_util.determine_tree_management_category(row["latvuskerros"])
        storey = vmi_util.determine_storey_for_tree(row["latvuskerros"])

        sapling = False
        tree_type = vmi2internal.convert_tree_type(row["tree_type"])

        damage_raw = row["tuhon_ilmiasu"]
        damage_type = None if damage_raw in ("  ", " ", ".", "") else damage_raw.strip()
        crown_class = vmi2internal.convert_crown_class(row["latvuskerros"])

        trees.identifier[i] = identifier
        if tree_number is not None:
            trees.tree_number[i] = tree_number
        trees.species[i] = species
        trees.breast_height_diameter[i] = breast_height_diameter
        if height is not None:
            trees.height[i] = height
        if measured_height is not None:
            trees.measured_height[i] = measured_height
        if breast_height_age is not None:
            trees.breast_height_age[i] = breast_height_age
        if biological_age is not None:
            trees.biological_age[i] = biological_age
        trees.stems_per_ha[i] = stems_per_ha
        if origin is not None:
            trees.origin[i] = origin
        trees.management_category[i] = management_category
        if tree_category is not None:
            trees.tree_category[i] = tree_category
        if storey is not None:
            trees.storey[i] = storey
        trees.sapling[i] = sapling
        if tree_type is not None:
            trees.tree_type[i] = tree_type
        if damage_type is not None:
            trees.damage_type[i] = damage_type
        if crown_class is not None:
            trees.crown_class[i] = crown_class
