from typing import Generator, override

from lukefi.metsi.data.conversion import vmi2internal
from lukefi.metsi.data.enums.internal import Origin, CRS
from lukefi.metsi.data.enums.vmi import VmiIteration
from lukefi.metsi.data.formats import util
from lukefi.metsi.data.formats.declarative_conversion import Conversion
from lukefi.metsi.data.formats.forest_builder_base import RowKind, VMIBuilder
from lukefi.metsi.data.formats.nfi.vmi_const import (
    VMI11_COUNTY_AREAS,
    VMI11_STAND_INDICES,
    VMI11_STRATUM_INDICES,
    VMI11_TREE_INDICES
)
from lukefi.metsi.data.formats.nfi import vmi_util
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.core.exceptions import MetsiException


class VMI11Builder(VMIBuilder):

    def __init__(self,
                 builder_flags: dict[str, bool],
                 declared_conversions: dict[str, Conversion],
                 data_rows: Generator[str]) -> None:
        super().__init__(builder_flags, declared_conversions)
        for row in data_rows:
            kind = self._classify_row(row)

            if kind == RowKind.STAND:
                self.stand_rows.append(vmi_util.generate_source_data(VMI11_STAND_INDICES, row))

            elif kind == RowKind.STRATUM and self.builder_flags.get("strata", False):
                stratum_row = vmi_util.generate_source_data(VMI11_STRATUM_INDICES, row)
                stand_identifier = vmi_util.generate_stand_identifier(stratum_row)
                self.stratum_rows.setdefault(stand_identifier, []).append(stratum_row)

            elif kind == RowKind.TREE and self.builder_flags.get("measured_trees", False):
                tree_row = vmi_util.generate_source_data(VMI11_TREE_INDICES, row)
                stand_identifier = vmi_util.generate_stand_identifier(tree_row)
                self.tree_rows.setdefault(stand_identifier, []).append(tree_row)

        if len(self.stand_rows) == 0:
            raise MetsiException("Source data did not contain any valid VMI11 stand rows")

    @staticmethod
    def _classify_row(row: str) -> RowKind:
        try:
            return RowKind(row[VMI11_STAND_INDICES["row_type"]])
        except ValueError:
            return RowKind.UNKNOWN

    @override
    def build(self) -> list[ForestStand]:
        result: dict[str, ForestStand] = {}

        for i, row in enumerate(self.stand_rows):
            try:
                stand = self._convert_stand_entry(row, i + 1)
                result[stand.identifier] = stand
            except Exception as e:
                raise MetsiException(f"Parsing stand row {row} failed: {e}") from e

        if self.builder_flags.get("strata", False):
            for stand_identifier, stand_stratum_rows in self.stratum_rows.items():
                strata = TreeStrata(len(stand_stratum_rows))
                for j, stand_stratum_row in enumerate(stand_stratum_rows):
                    try:
                        self._convert_stratum_entry(strata, stand_stratum_row, j)
                    except Exception as e:
                        raise MetsiException(f"Parsing stratum row {stand_stratum_row} failed: {e}") from e

                result[stand_identifier].tree_strata = strata

        if self.builder_flags.get("measured_trees", False):
            for stand_identifier, stand_tree_rows in self.tree_rows.items():
                stand_ = result.get(stand_identifier)
                if stand_ is None:
                    continue

                trees = ReferenceTrees(len(stand_tree_rows))
                for j, stand_tree_row in enumerate(stand_tree_rows):
                    try:
                        self._convert_tree_entry(trees, stand_tree_row, j, stand_.forestry_centre_id, stand_.ahvkeilaus)
                    except Exception as e:
                        raise MetsiException(f"Parsing tree row {stand_tree_row} failed: {e}") from e

                result[stand_identifier].reference_trees = trees

        return list(result.values())

    def _convert_stand_entry(self, row: dict[str, str], stand_id: int | None = None) -> ForestStand:
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

        result.municipality_id = int(row["municipality"])

        result.year = vmi_util.parse_date(row["date"]).year
        result.start_year = result.year
        result.development_class = vmi2internal.convert_development_class(row["kehitysluokka"])
        result.main_tree_species_dominant_storey = vmi_util.determine_main_tree_species_dominant_storey(
            row["main_tree_species_dominant_storey"],
            result.site_type_category,
        )

        result.basal_area = util.parse_type(row["pohjapintaala"], float)
        result.region = None

        area_ha = VMI11Builder._determine_area_ha(
            vmi_util.parse_forestry_centre(row["forestry_centre"]),
            int(row["lohkomuoto"]),
            row["area_ha"],
            inventointitunnus=row["inventointitunnus"],
            lohy_raw=row["section_y"],
            ahvkeilaus=row["ahvkeilaus"],
        )
        result.set_area(area_ha)
        result.ahvkeilaus = row["ahvkeilaus"]

        result.area_weight_factors = vmi_util.determine_area_factors(
            row["osuus7m"],
            row["osuusrel"],
        )

        lat = util.get_or_default(util.parse_type(row["lat_measured"], float), 0.0)
        lon = util.get_or_default(util.parse_type(row["lon_measured"], float), 0.0)
        if not lat:
            lat = util.get_or_default(util.parse_type(row["lat"], float), 0.0)
        if not lon:
            lon = util.get_or_default(util.parse_type(row["lon"], float), 0.0)

        height = VMI11Builder._transform_height_above_sea_level(row["height_above_sea_level"])
        result.set_geo_location(lat, lon, height, CRS.EPSG_2393)

        result.drainage_year = vmi_util.determine_drainage_year(row["ojitus_aika"], result.year)
        result.soil_surface_preparation_year = vmi_util.determine_soil_surface_preparation_year(
            row["maanmuokkaus_aika"],
            result.year,
        )
        result.regeneration_area_cleaning_year = vmi_util.determine_clearing_of_reform_sector_year(
            row["muu_toimenpide"],
            row["muu_toimenpide_aika"],
            result.year,
        )
        result.artificial_regeneration_year = vmi_util.determine_artificial_regeneration_year(
            row["viljely"],
            row["viljely_aika"],
            result.year,
        )

        maintenance_details = vmi2internal.convert_forest_maintenance_details(
            row["hakkuu_tapa"],
            row["hakkuu_aika"],
            result.year)
        result.young_stand_tending_year = maintenance_details[0]
        result.cutting_year = maintenance_details[1]
        result.method_of_last_cutting = maintenance_details[2]

        result.ds_main_tree_species_biological_age = VMI11Builder._determine_dominant_storey_age(
            row["vallitsevanjakson_d13ika"],
            row["vallitsevanjakson_ikalisays"],
        )

        result.peatland_type = vmi2internal.convert_peatland_forest_type(row["suotyy"])
        result.drained_peatland_type = vmi2internal.convert_drained_peatland_forest_type(row["tkgtyy"])
        result.under_storey = bool(util.parse_type(row["alikehl"], int))
        result.over_storey = bool(util.parse_type(row["ylikehl"], int))

        if result.land_use_category and result.forestry_centre_id is not None and result.owner_category is not None:
            is_ahvenanmaa = result.forestry_centre_id == 0
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

        result = self.conversion_reader.apply_conversions(result, row)

        return result

    @staticmethod
    def _determine_area_ha(forestry_centre: int,
                           lohkomuoto: int,
                           eduala_raw: str,
                           inventointitunnus: str | None = None,
                           lohy_raw: str | None = None,
                           ahvkeilaus: str | None = None) -> float:
        """
        Determine stand area_ha for VMI11.

        Default:
        - area_ha = eduala with 5 decimals

        Lookup:
        - Normal lohkomuoto osite in {1,2,3,4} -> lookup from VMI11_COUNTY_AREAS
        - Ahvenanmaa special osite in {300,400} -> lookup from VMI11_COUNTY_AREAS (metkes=0 row)

        Ahvenanmaa rules:
        - inventointitunnus = P and ahvkeilaus = A => osite 300 => area 100.39
        - inventointitunnus = P and ahvkeilaus = B => osite 400 => area 148.78
        - inventointitunnus = K and lohy(1:1) = 3 => osite 300 => area 100.39
        - inventointitunnus = K and lohy(1:1) = 4 => osite 400 => area 148.78
        """
        default_area_ha = vmi_util.parse_vmi_area_ha(eduala_raw)

        inv = (inventointitunnus or "").strip().upper()
        ak = (ahvkeilaus or "").strip().upper()
        lohy_first = ((lohy_raw or "").strip()[:1] or "")

        if inv == "P" and ak in ("A", "B"):
            osite = 300 if ak == "A" else 400
            return round(VMI11Builder._get_area_ha(0, osite), 4)

        if inv == "K" and lohy_first in ("3", "4"):
            osite = 300 if lohy_first == "3" else 400
            return round(VMI11Builder._get_area_ha(0, osite), 4)

        osite = lohkomuoto
        use_lookup = (1 <= osite <= 4) or (forestry_centre == 0)

        if not use_lookup:
            return round(default_area_ha, 4)

        try:
            return round(VMI11Builder._get_area_ha(forestry_centre, osite), 4)
        except KeyError as exc:
            raise MetsiException(
                f"No area_ha lookup value for VMI11: metkes={forestry_centre}, osite={osite}. "
                f"inventointitunnus={inv!r}, lohy={lohy_raw!r}, ahvkeilaus={ak!r}, lohkomuoto={lohkomuoto}."
            ) from exc

    @staticmethod
    def _get_area_ha(forestry_centre: int, osite: int) -> float:
        """
        VMI11 area lookup.
        Returns area_ha for given metkes (forestry_centre) and osite (lohkomuoto / 300 / 400).
        """
        try:
            return VMI11_COUNTY_AREAS[forestry_centre][osite]
        except KeyError as exc:
            raise KeyError(f"No VMI11 area_ha for metkes={forestry_centre}, osite={osite}") from exc

    @staticmethod
    def _transform_height_above_sea_level(sourcevalue: str) -> float | None:
        """
        Transform given number value string from desimeters to meters.
        Returning float, or None on error.
        """
        try:
            return float(sourcevalue) / 10.0
        except ValueError:
            return None

    @staticmethod
    def _determine_dominant_storey_age(ds_bh_age: str, ds_age_increase: str) -> float:
        """ Dominant storey age is composed of dominant storey breast height age and age increase for vmi12. """
        a = util.get_or_default(vmi_util.parse_float(ds_bh_age), 0.0)
        b = util.get_or_default(vmi_util.parse_float(ds_age_increase), 0.0)
        return a + b

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
        stratum_rank = vmi2internal.convert_stratum_rank(row["stratum_rank"])
        storey = vmi_util.determine_storey_for_stratum(stratum_rank)

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
        strata.storey[i] = storey
        strata.sapling_stems_per_ha[i] = sapling_stems_per_ha
        strata.stratum_rank[i] = stratum_rank

    @staticmethod
    def _convert_tree_entry(trees: ReferenceTrees,
                            row: dict[str, str],
                            i: int,
                            forestry_centre_id: int | None,
                            ahvkeilaus: str | None,
                            height_conversion_factor: float = 10.0,
                            measured_height_conversion_factor: float = 10.0):
        identifier = vmi_util.generate_tree_identifier(row)
        tree_number = util.parse_type(row["tree_number"], int)
        species = vmi2internal.convert_species(row["species"])
        tree_category = vmi2internal.convert_tree_category(row["tree_category"])

        breast_height_diameter = util.get_or_default(util.parse_float(row["diameter"]), 0.0)
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
                                                       vmi_version=VmiIteration.VMI11,
                                                       forestry_centre_id=forestry_centre_id,
                                                       ahvkeilaus=ahvkeilaus)
        origin = Origin.NATURAL

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
        trees.stems_per_ha[i] = stems_per_ha
        trees.sapling[i] = sapling
        trees.management_category[i] = management_category

        if height is not None:
            trees.height[i] = height

        if measured_height is not None:
            trees.measured_height[i] = measured_height

        if breast_height_age is not None:
            trees.breast_height_age[i] = breast_height_age

        if biological_age is not None:
            trees.biological_age[i] = biological_age

        if origin is not None:
            trees.origin[i] = origin

        if tree_category is not None:
            trees.tree_category[i] = tree_category

        if storey is not None:
            trees.storey[i] = storey

        if tree_type is not None:
            trees.tree_type[i] = tree_type

        if damage_type is not None:
            trees.damage_type[i] = damage_type

        if crown_class is not None:
            trees.crown_class[i] = crown_class
