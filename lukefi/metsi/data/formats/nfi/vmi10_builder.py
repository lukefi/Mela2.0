from typing import Generator, override

import numpy as np

from lukefi.metsi.data.conversion import vmi2internal
from lukefi.metsi.data.enums.internal import Origin, Storey
from lukefi.metsi.data.enums.vmi import VmiIteration
from lukefi.metsi.data.formats import util
from lukefi.metsi.data.formats.declarative_conversion import Conversion
from lukefi.metsi.data.formats.forest_builder_base import RowKind, VMIBuilder
from lukefi.metsi.data.formats.nfi.vmi_const import VMI10_COUNTY_AREAS, VMI10_STAND_INDICES, VMI10_TREE_INDICES
from lukefi.metsi.data.formats.nfi import vmi_util
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.sim.utils import MetsiException


class VMI10Builder(VMIBuilder):

    def __init__(self,
                 builder_flags: dict[str, bool],
                 declared_conversions: dict[str, Conversion],
                 data_rows: Generator[str]) -> None:
        super().__init__(builder_flags, declared_conversions)
        for row in data_rows:
            kind = self._classify_row(row)

            if kind == RowKind.STAND:
                self.stand_rows.append(vmi_util.generate_source_data(VMI10_STAND_INDICES, row))

            elif kind == RowKind.TREE and self.builder_flags.get("measured_trees", False):
                tree_row = vmi_util.generate_source_data(VMI10_TREE_INDICES, row)
                stand_identifier = vmi_util.generate_stand_identifier(tree_row)
                self.tree_rows.setdefault(stand_identifier, []).append(tree_row)

        if len(self.stand_rows) == 0:
            raise MetsiException("Source data did not contain any valid VMI10 stand rows")

    @staticmethod
    def _classify_row(row: str) -> RowKind:
        row_type = row[VMI10_STAND_INDICES["row_type"]]
        if row_type == "1":
            return RowKind.STAND
        if row_type in ("2", "3"):
            return RowKind.TREE
        return RowKind.UNKNOWN

    @override
    def build(self) -> list[ForestStand]:
        result: dict[str, ForestStand] = {}

        for i, row in enumerate(self.stand_rows):
            try:
                stand = self._convert_stand_entry(row, i + 1)
                result[stand.identifier] = stand

                strata = TreeStrata(8)
                self._convert_strata_from_stand_entry(strata, row, stand.identifier, stand.basal_area or 0.0)
                stand.tree_strata = strata

            except Exception as e:
                raise MetsiException(f"Parsing stand row {row} failed: {e}") from e

        if self.builder_flags.get("measured_trees", False):
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

        parsed = vmi_util.parse_date(row["date"])
        if parsed is None:
            raise MetsiException("Year is None in VMI10 data")

        result.year = parsed.year
        result.start_year = parsed.year
        result.development_class = vmi2internal.convert_development_class(row["kehitysluokka"])

        area_ha = VMI10Builder._get_area_ha(
            vmi_util.parse_forestry_centre(row["forestry_centre"]),
            int(row["lohkomuoto"]),
        )
        result.set_area(area_ha)

        result.area_weight_factors = vmi_util.determine_area_factors(
            row["osuus7m"],
            row["osuusrel"],
        )

        result.basal_area = util.parse_type(row["basal_area"], float)
        result.drainage_year = vmi_util.determine_drainage_year(row["ojitus_aika"], result.year)
        result.fertilization_year = None

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

        lat = util.get_or_default(util.parse_type(row["lat_measured"], float), 0.0)
        lon = util.get_or_default(util.parse_type(row["lon_measured"], float), 0.0)
        if not lat:
            lat = util.get_or_default(util.parse_type(row["lat"], float), 0.0)
        if not lon:
            lon = util.get_or_default(util.parse_type(row["lon"], float), 0.0)
        height_dm = util.get_or_default(util.parse_type(row["height_above_sea_level"], float), 0.0)
        result.set_geo_location(lat, lon, height_dm / 10.0, "EPSG:2393")

        result.soil_surface_preparation_year = vmi_util.determine_soil_surface_preparation_year(
            row["maanmuokkaus_aika"],
            result.year
        )
        result.peatland_type = vmi2internal.convert_peatland_forest_type(row["suotyy"])
        result.drained_peatland_type = vmi2internal.convert_drained_peatland_forest_type(row["tkgtyy"])

        result.region = None
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
    def _get_area_ha(forestry_centre: int, lohkomuoto: int) -> float:
        """
        VMI10 area lookup.
        Returns area_ha for given forestry_centre and lohkomuoto.
        """
        try:
            return VMI10_COUNTY_AREAS[forestry_centre][lohkomuoto]
        except KeyError as exc:
            raise KeyError(f'No VMI10 area_ha for keskus={forestry_centre}, lohkomuoto={lohkomuoto}') from exc

    @staticmethod
    def _convert_strata_from_stand_entry(strata: TreeStrata,
                                         row: dict[str, str],
                                         stand_identifier: str,
                                         stand_basal_area: float):
        fallback_age = vmi_util.parse_float0(row["metsikon_ika"])

        jakso2_ppa = vmi_util.parse_float0(row["jakso2_ppa"])
        jakso1_ppa = max(0.0, stand_basal_area - jakso2_ppa)

        running_numb = 0

        for seg_no in (1, 2):
            ppa_total = jakso1_ppa if seg_no == 1 else jakso2_ppa

            asema = row[f"jakso{seg_no}_asema"]
            synty = row[f"jakso{seg_no}_syntytapa"]
            stems1000 = row[f"jakso{seg_no}_kokonaisrunkoluku1000"]
            d_cm = row[f"jakso{seg_no}_keskilapimitta_cm"]
            h_dm = row[f"jakso{seg_no}_keskipituus_dm"]
            d13ika = row[f"jakso{seg_no}_d13ika"]
            ikalis = row[f"jakso{seg_no}_ikalisays"]

            VMI10Builder._emit_stratum(strata,
                                       stand_identifier,
                                       row[f"jakso{seg_no}_paapuulaji"],
                                       row[f"jakso{seg_no}_paapuulaji_osuus"],
                                       stems1000, d_cm, h_dm, d13ika, ikalis, synty, asema, ppa_total,
                                       fallback_age,
                                       running_numb
                                       )
            running_numb += 1
            for j in (1, 2, 3):
                VMI10Builder._emit_stratum(strata,
                                           stand_identifier,
                                           row[f"jakso{seg_no}_sivulaji{j}"],
                                           row[f"jakso{seg_no}_sivulaji{j}_osuus"],
                                           stems1000, d_cm, h_dm, d13ika, ikalis, synty, asema, ppa_total,
                                           fallback_age,
                                           running_numb
                                           )
                running_numb += 1

        missing_indices = np.nonzero(strata.identifier == "")[0]
        strata.delete(missing_indices)

    @staticmethod
    def _emit_stratum(strata: TreeStrata,
                      stand_identifier: str,
                      species_code_raw: str,
                      share_raw: str,
                      seg_stems1000_raw: str,
                      seg_d_cm_raw: str,
                      seg_h_dm_raw: str,
                      seg_d13_age_raw: str,
                      seg_age_inc_raw: str,
                      seg_syntytapa_raw: str,
                      seg_asema_raw: str,
                      seg_ppa_total: float,
                      fallback_age: float,
                      i: int):
        species_code = (species_code_raw or "").strip()
        if not species_code or species_code in (".", "0"):
            return
        share = vmi_util.parse_share_tenths(share_raw)
        if share <= 0.0:
            return

        species = vmi2internal.convert_species(species_code)

        basal_area = seg_ppa_total * share
        stems_per_ha = vmi_util.parse_float0(seg_stems1000_raw) * 1000.0 * share
        mean_diameter = vmi_util.parse_float0(seg_d_cm_raw)
        mean_height = vmi_util.parse_float0(seg_h_dm_raw) / 10.0
        age = VMI10Builder._segment_age_years(seg_d13_age_raw, seg_age_inc_raw, fallback_age=fallback_age)

        syntytapa = VMI10Builder._determine_stratum_origin(seg_syntytapa_raw)
        storey = VMI10Builder._determine_storey_for_segment(seg_asema_raw)

        identifier = f"{stand_identifier}-{i + 1}-stratum"

        strata.identifier[i] = identifier
        strata.species[i] = species
        strata.mean_diameter[i] = mean_diameter
        strata.mean_height[i] = mean_height
        strata.breast_height_age[i] = vmi_util.parse_int0(seg_d13_age_raw)
        strata.biological_age[i] = age
        strata.stems_per_ha[i] = stems_per_ha
        strata.basal_area[i] = basal_area
        strata.origin[i] = syntytapa
        strata.stratum_number[i] = i + 1
        strata.storey[i] = storey
        strata.sapling_stems_per_ha[i] = 0.0
        strata.number_of_generated_trees[i] = 0

    @staticmethod
    def _segment_age_years(d13_age_raw: str, age_inc_raw: str, fallback_age: float) -> float:
        a = vmi_util.parse_float0(d13_age_raw)
        b = vmi_util.parse_float0(age_inc_raw)
        age = a + b
        return fallback_age if age <= 0.0 else age

    @staticmethod
    def _determine_stratum_origin(source_origin: str) -> Origin:
        if source_origin in ("0", "1", "2"):
            return Origin.NATURAL
        if source_origin == "3":
            return Origin.PLANTED
        if source_origin == "4":
            return Origin.SEEDED
        return Origin.NATURAL

    @staticmethod
    def _determine_storey_for_segment(asema_raw: str) -> Storey:
        """
        VMI10 jakson asema:
        1 Vallitseva jakso
        2 Ylispuusto
        3 Jättöylispuusto
        4 Verhopuusto
        5 Kehityskelpoinen alikasvos
        6 Kehityskelvoton alikasvos
        7 Vaihtuva taimiaines
        """
        v = (asema_raw or "").strip()
        if not v or v == ".":
            return Storey.INDETERMINATE
        if v == "1":
            return Storey.DOMINANT
        if v == "2":
            return Storey.OVER
        if v == "3":
            return Storey.REMOVAL
        if v == "4":
            return Storey.SPARE
        if v == "5":
            return Storey.UNDER
        if v == "6":
            return Storey.REMOTE
        if v == "7":
            return Storey.INDETERMINATE
        return Storey.INDETERMINATE

    @staticmethod
    def _convert_tree_entry(trees: ReferenceTrees,
                            row: dict[str, str],
                            i: int,
                            forestry_centre_id: int | None):
        identifier = vmi_util.generate_tree_identifier(row)
        tree_number = vmi_util.get_or_default(vmi_util.parse_type(row["tree_number"], int), 0)

        species = vmi2internal.convert_species(row["species"])

        raw_tc = row["tree_category"].strip()
        tc_enum = vmi2internal.convert_tree_category(raw_tc)

        tree_category = tc_enum.value if tc_enum else None
        breast_height_diameter = vmi_util.transform_tree_diameter(row["diameter"])

        # Heights are in dm -> meters via /10
        height = vmi_util.determine_tree_height(row["height"], conversion_factor=10.0)

        breast_height_age = vmi_util.parse_type(row["d13_age"], float)
        biological_age = vmi_util.parse_type(row["total_age"], float)

        stems_per_ha = vmi_util.determine_stems_per_ha(breast_height_diameter, vmi_version=VmiIteration.VMI10,
                                                       forestry_centre_id=forestry_centre_id)

        management_category = vmi_util.determine_tree_management_category(row["latvuskerros"])
        storey = vmi_util.determine_storey_for_tree(row["latvuskerros"])
        tree_type = vmi2internal.convert_tree_type(row["tree_type"])

        tuhon_raw = row["tuhon_ilmiasu"]
        damage_type = None if tuhon_raw in (" ", ".", "") else tuhon_raw.strip()

        trees.identifier[i] = identifier
        trees.tree_number[i] = tree_number
        trees.species[i] = species
        if tree_category is not None:
            trees.tree_category[i] = tree_category
        trees.breast_height_diameter[i] = breast_height_diameter
        if height is not None:
            trees.height[i] = height
        if breast_height_age is not None:
            trees.breast_height_age[i] = breast_height_age
        if biological_age is not None:
            trees.biological_age[i] = biological_age
        trees.stems_per_ha[i] = stems_per_ha
        trees.origin[i] = 0
        trees.management_category[i] = management_category
        if storey is not None:
            trees.storey[i] = storey
        trees.sapling[i] = False
        if tree_type is not None:
            trees.tree_type[i] = tree_type
        if damage_type is not None:
            trees.damage_type[i] = damage_type
