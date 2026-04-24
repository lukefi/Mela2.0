from typing import override

import numpy as np

from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.conversion import vmi2internal
from lukefi.metsi.data.enums.internal import Origin, Storey
from lukefi.metsi.data.enums.vmi import VmiIteration
from lukefi.metsi.data.formats import util
from lukefi.metsi.data.formats.declarative_conversion import Conversion
from lukefi.metsi.data.formats.forest_builder_base import RowKind, VMIBuilder
from lukefi.metsi.data.formats.nfi.vmi_const import (
    VMI9_STAND_COMMON,
    VMI9_STAND_INDICES_ESUOMI,
    VMI9_STAND_INDICES_PSUOMI,
    VMI9_TREE_INDICES)
from lukefi.metsi.data.formats.nfi import vmi_util
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata


class VMI9Builder(VMIBuilder):
    def __init__(self,
                 builder_flags: dict[str, bool],
                 declared_conversions: dict[str, Conversion],
                 data_rows: list[str]) -> None:
        super().__init__(builder_flags, declared_conversions)
        for row in data_rows:
            kind = self._classify_row(row)

            if kind == RowKind.STAND:
                self.stand_rows.append(vmi_util.generate_source_data(self._select_stand_indices(row), row))

            elif kind == RowKind.TREE and self.builder_flags.get("measured_trees", False):
                tree_row = vmi_util.generate_source_data(VMI9_TREE_INDICES, row)
                stand_identifier = vmi_util.generate_stand_identifier(tree_row)
                self.tree_rows.setdefault(stand_identifier, []).append(tree_row)

    @staticmethod
    def _classify_row(row: str) -> RowKind:
        row_type = row[VMI9_STAND_COMMON["row_type"]]
        if row_type == "1":
            return RowKind.STAND
        if row_type == "2":
            return RowKind.TREE
        return RowKind.UNKNOWN

    @staticmethod
    def _select_stand_indices(row: str) -> dict[str, slice]:
        """
        VMI9 has two fixed-width layouts (Etelä-Suomi and Pohjois-Suomi).
        metsäkeskusjako (metkes):
        0..10 => Etelä-Suomi
        11..13 => Pohjois-Suomi
        """

        forestry_centre = util.parse_int(row[VMI9_STAND_COMMON["forestry_centre"]].strip())
        if forestry_centre is None:
            raise MetsiException("Forestry_centre information is missing")

        return VMI9_STAND_INDICES_PSUOMI if 11 <= forestry_centre <= 13 else VMI9_STAND_INDICES_ESUOMI

    @override
    def build(self) -> list[ForestStand]:
        result: dict[str, ForestStand] = {}

        for i, row in enumerate(self.stand_rows):
            try:
                stand = self._convert_stand_entry(row, i + 1)
                result[stand.identifier] = stand

                strata = TreeStrata(6)
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

        parsed = vmi_util.parse_date(row["date"])
        if parsed is None:
            raise MetsiException("Year is None in VMI9 data")

        result.year = parsed.year
        result.start_year = parsed.year
        result.development_class = vmi2internal.convert_development_class(row["kehitysluokka"])

        area_ha = vmi_util.parse_vmi_area_ha(row["area_ha"])
        result.set_area(area_ha)

        result.area_weight_factors = vmi_util.determine_area_factors(
            row["osuus7m"],
            row["osuusrel"],
        )

        result.drainage_year = vmi_util.determine_drainage_year(
            row["ojitus_aika"],
            result.year
        )
        result.region = None
        result.basal_area = util.parse_type(row["basal_area"], float)
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

        result.forest_management_category = VMI9Builder._determine_forest_management_category(row)

        return result

    @staticmethod
    def _determine_forest_management_category(row: dict[str, str]) -> float:
        """
        VMI9 käsittelyluokka (forest_management_category) calculation.
        """

        owner = vmi_util.parse_int0(row["owner_group"])
        ptraj = vmi_util.parse_int0(row["ptraj"])
        pttark = vmi_util.parse_int0(row["pttark"])
        land_category = vmi_util.parse_int0(row["land_category"])

        ml123_ala = vmi_util.parse_float0(row["ml123ala"])

        abt1 = vmi_util.parse_int0(row["abi1kasehd"])
        abt1_ala = vmi_util.parse_float0(row["abi1ala"])

        abt2 = vmi_util.parse_int0(row["abi2kasehd"])
        abt2_ala = vmi_util.parse_float0(row["abi2ala"])

        abt3 = vmi_util.parse_int0(row["abi3kasehd"])
        abt3_ala = vmi_util.parse_float0(row["abi3ala"])

        # mhptrajtar is only meaningful in North Finland; in South your indices are slice(0,0) or value is blank.
        mhtark = vmi_util.parse_int0(row["mhptrajtar"]) if "mhptrajtar" in row else 0

        # ------------------------------------------------------------
        # Start
        k = 1.0

        # --- avainbiotoopit ---
        avainbt = max(abt1, abt2, abt3)

        avainbt_pinta_ala = 0.0
        avainbt_ala = 0.0

        if avainbt == 2:
            avainbt_pinta_ala = abt1_ala + abt2_ala + abt3_ala

        if ml123_ala > 0.0:
            avainbt_ala = avainbt_pinta_ala / ml123_ala

        # --- metsähallituksen rajoituksen tarkennus ---
        if owner == 4:
            ptraj = 0

        if owner == 4 and mhtark in (2, 3, 4, 5, 6, 9):
            ptraj = mhtark

        # --- käsittelyluokka ---
        if ptraj in (101, 102):
            k = 7.1

        if ptraj in (401, 402, 403, 404, 408, 409, 410):
            k = 7.2

        if ptraj == 105 and owner not in (4, 5):
            k = 7.3

        if ptraj == 105 and owner in (4, 5):
            k = 7.5

        if ptraj in (104, 106):
            k = 7.5

        if ptraj == 103:
            k = 7.4

        if ptraj in (201, 205, 301):
            k = 7.5

        if ptraj == 405:
            k = 2.1

        if ptraj in (202, 203, 307):
            k = 2.4

        if ptraj == 304:
            k = 2.2

        if ptraj == 303:
            k = 2.3

        if ptraj == 501 and pttark in (1, 2, 3, 4, 5) and owner in (4, 5):
            k = 2.4

        if ptraj in (309, 306, 302):
            k = 2.4

        if ptraj == 107 and pttark == 1:
            k = 7.5

        if ptraj == 107 and pttark != 1:
            k = 2.4

        if ptraj == 606 and pttark == 1 and avainbt == 2:
            k = 7.5

        if ptraj == 606 and pttark != 1 and avainbt == 2:
            k = 2.4

        if ptraj == 608 and pttark == 1 and avainbt == 2 and avainbt_ala >= 0.5:
            k = 7.5

        if ptraj == 608 and pttark != 1 and avainbt == 2 and avainbt_ala >= 0.5:
            k = 2.4

        if ptraj == 601 and owner == 4:
            k = 2.4

        if ptraj == 305:
            k = 2.4

        if owner == 4 and ptraj == 0:
            k = 1.0

        if owner == 4 and ptraj in (2, 3, 4, 5):
            k = 2.4

        if owner == 4 and ptraj in (6, 9):
            k = 7.5

        # --- maaluokka adjustments ---
        if land_category == 2 and k == 1.0:
            k = 3.5

        if land_category == 2 and 2.0 <= k < 3.0:
            k = k + 1.0

        if land_category == 3 and k == 1.0:
            k = 6.5

        if land_category == 3 and 2.0 <= k < 3.0:
            k = k + 4.0

        if land_category == 3 and k > 7.0:
            k = 7.6

        return float(k)

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

            main_share_raw = row[f"jakso{seg_no}_paapuulaji_osuus"]
            siv1_share_raw = row[f"jakso{seg_no}_sivulaji1_osuus"]

            main_t = vmi_util.parse_int0(main_share_raw)
            siv1_t = vmi_util.parse_int0(siv1_share_raw)

            siv2_t = max(0, 10 - main_t - siv1_t)

            VMI9Builder._emit_stratum(
                strata,
                stand_identifier,
                row[f"jakso{seg_no}_paapuulaji"],
                str(main_t),
                stems1000, d_cm, h_dm, d13ika, ikalis, synty, asema, ppa_total,
                fallback_age,
                running_numb
            )
            running_numb += 1

            VMI9Builder._emit_stratum(
                strata,
                stand_identifier,
                row[f"jakso{seg_no}_sivulaji1"],
                str(siv1_t),
                stems1000, d_cm, h_dm, d13ika, ikalis, synty, asema, ppa_total,
                fallback_age,
                running_numb
            )
            running_numb += 1

            VMI9Builder._emit_stratum(
                strata,
                stand_identifier,
                row[f"jakso{seg_no}_sivulaji2"],
                str(siv2_t),
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
        mean_height = vmi_util.parse_float0(seg_h_dm_raw) / 10.0  # dm -> m
        age = VMI9Builder._segment_age_years(seg_d13_age_raw, seg_age_inc_raw, fallback_age=fallback_age)

        syntytapa = VMI9Builder._determine_stratum_origin(seg_syntytapa_raw)
        storey = VMI9Builder._determine_storey_for_segment(seg_asema_raw)

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
        if source_origin in ("1", "2"):
            return Origin.NATURAL
        if source_origin in ("3", "5", "7", "8"):
            return Origin.PLANTED
        if source_origin in ("4", "6"):
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
        tree_number = util.get_or_default(vmi_util.parse_type(row["tree_number"], int), 0)

        species = vmi2internal.convert_species(row["species"])

        raw_tc = row["tree_category"].strip()
        tc_enum = vmi2internal.convert_tree_category(raw_tc)

        tree_category = tc_enum.value if tc_enum else None
        breast_height_diameter = vmi_util.transform_tree_diameter(row["diameter"])

        # dm -> m
        height = vmi_util.determine_tree_height(row["height"], conversion_factor=10.0)

        breast_height_age, biological_age = vmi_util.determine_tree_age_values(
            row["d13_age"],
            row["age_increase"],
            row["total_age"],
        )

        stems_per_ha = vmi_util.determine_stems_per_ha(breast_height_diameter, vmi_version=VmiIteration.VMI9,
                                                       forestry_centre_id=forestry_centre_id)

        management_category = vmi_util.determine_tree_management_category(row["latvuskerros"])
        storey = vmi_util.determine_storey_for_tree(row["latvuskerros"])

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
        if damage_type is not None:
            trees.damage_type[i] = damage_type
