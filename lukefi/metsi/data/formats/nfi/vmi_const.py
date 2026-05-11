""" Constant data indices of the app """

VMI9_STAND_COMMON: dict[str, slice] = {

    "lohkomuoto": slice(8, 9),         # lohmuo  (col 9)
    "section_y": slice(2, 5),          # lohy    (col 3-5)
    "section_x": slice(5, 8),          # lohx    (col 6-8)
    "test_area_number": slice(10, 12),  # koeala  (col 11-12)
    "stand_number": slice(12, 13),     # kuvio   (col 13)
    "row_type": slice(13, 14),         # tielaji (col 14)
    "lat": slice(18, 25),              # pkoonim (col 19-25)
    "lon": slice(25, 32),              # ikoonim (col 26-32)
    "height_above_sea_level": slice(82, 87),  # korkeus dm (col 83-87)
    "area_ha": slice(50, 59),          # eduala (col 51-59)
    "osuusrel": slice(36, 38),         # osuusrel (col 37-38)
    "osuus7m": slice(59, 61),          # osuus7m (col 60-61)
    "forestry_centre": slice(66, 68),  # metkes (col 67-68)
    "municipality": slice(68, 71),     # kunta (col 69-71)
    "owner_group": slice(81, 82),      # omiryh (col 82)
    "degree_days": slice(87, 91),      # lamsum1 (col 88-91)
    "ptraj": slice(93, 96),             # ptraj (col 96)
    "pttark": slice(96, 97),             # pttark (col 97)
    "suojametsakoodi": slice(97, 98),   # suojamet (col 98)
    "fra_class": slice(99, 100),       # fraluo (col 100) in PSUOMI, blank in ESUOMI
    "land_category": slice(100, 101),  # maaluo (col 101)
    "land_category_detail": slice(101, 102),  # maaluotar (col 102)
    "paatyyppi": slice(104, 105),      # paatyy (col 105)
    "kasvupaikkatunnus": slice(106, 107),  # kaspai (col 107)
    "ojitus_tilanne": slice(126, 127),  # ojitil (col 127)
    "ojitus_aika": slice(128, 129),     # ojiaik (col 129)
    "ojitus_tarve": slice(129, 130),    # ojitar (col 130)
    "tax_class": slice(132, 133),       # veroluo (col 133)
    "tax_class_reduction": slice(133, 134),  # verotar (col 134)
    "ppa1": slice(203, 206),           # ppa1 (col 204-206)
    "ppa2": slice(207, 210),           # ppa2 (col 208-210)
    "ppa3": slice(211, 214),           # ppa3 (col 212-214)
    "ppa4": slice(0, 0),               # not present in VMI9
    "ppa5": slice(0, 0),               # not present in VMI9
    "metsikon_ika": slice(184, 187),
    "jakso1_asema": slice(140, 141),                    # jakasema 141
    "kehitysluokka": slice(141, 142),                   # kehluo
    "jakso1_syntytapa": slice(142, 143),                # syntapa 143
    "jakso1_kokonaisrunkoluku1000": slice(159, 161),    # rlkok 160-161
    "jakso1_keskipituus_dm": slice(163, 166),           # keskipit 164-166
    "jakso1_keskilapimitta_cm": slice(167, 169),        # keskilpm 168-169
    "jakso1_d13ika": slice(169, 172),                   # d13ika 170-172
    "jakso1_ikalisays": slice(173, 175),                # ikalis 174-175

    "jakso1_paapuulaji": slice(143, 145),               # paaplaji 144-145
    "jakso1_paapuulaji_osuus": slice(145, 147),         # paaplajios 146-147
    "jakso1_sivulaji1": slice(147, 149),                # sivplaji1 148-149
    "jakso1_sivulaji1_osuus": slice(149, 151),          # sivplaji1os 150-151
    "jakso1_sivulaji2": slice(151, 153),                # sivplaji2 152-153
    "maanmuokkaus_aika": slice(194, 194),               # maakasaika
    "basal_area": slice(215, 217),                      # kuvppa, kuvion ppa
    "jakso2_ppa": slice(217, 219),                      # j2ppa 218-219
    "jakso2_asema": slice(220, 221),                    # j2asema 221
    "jakso2_syntytapa": slice(222, 223),                # j2syntapa 223
    "jakso2_kokonaisrunkoluku1000": slice(239, 241),    # j2rlkok 240-241
    "jakso2_keskipituus_dm": slice(243, 246),           # j2keskipit 244-246
    "jakso2_keskilapimitta_cm": slice(247, 249),        # j2keskilpm 248-249
    "jakso2_d13ika": slice(249, 252),                   # j2d13ika 250-252
    "jakso2_ikalisays": slice(253, 255),                # j2ikalis 254-255
    "jakso2_paapuulaji": slice(223, 225),               # j2paaplaji 224-225
    "jakso2_paapuulaji_osuus": slice(225, 227),         # j2paaplajios 226-227
    "jakso2_sivulaji1": slice(227, 229),                # j2sivplaji1 228-229
    "jakso2_sivulaji1_osuus": slice(229, 231),          # j2sivplaji1os 230-231
    "jakso2_sivulaji2": slice(231, 233),                # j2sivplaji2 232-233
    "ml123ala": slice(270, 272),                    # Maaluokkien 1-3 pinta-ala
    "abi1kasehd": slice(275, 276),                 # 276	kasittelyehdotus
    "abi1ala": slice(276, 278),                    # 278	pinta-ala (aaria)

    "abi2kasehd": slice(281, 282),                 # 282	kasittelyehdotus
    "abi2ala": slice(282, 284),                    # 284	pinta-ala (aaria)

    "abi3kasehd": slice(287, 288),                 # 288	kasittelyehdotus
    "abi3ala": slice(288, 290),                    # 290	pinta-ala (aaria)

    "mhptrajtar": slice(305, 306),                    # MH:n rajoituksen tarkennus (Pohjois-Suomi)

    "lat_measured": slice(306, 313),
    "lon_measured": slice(313, 320),

}

# Stand row indices – Etelä-Suomi (pvm at 45-50)
VMI9_STAND_INDICES_ESUOMI = dict(VMI9_STAND_COMMON)
VMI9_STAND_INDICES_ESUOMI["date"] = slice(44, 50)  # pvm (col 45-50)
VMI9_STAND_INDICES_ESUOMI["mhptrajtar"] = slice(0, 0)  # Not in use

# Stand row indices – Pohjois-Suomi (pvm at 76-81)
VMI9_STAND_INDICES_PSUOMI = dict(VMI9_STAND_COMMON)
VMI9_STAND_INDICES_PSUOMI["date"] = slice(75, 81)  # pvm (col 76-81)


# Tree row indices
VMI9_TREE_INDICES: dict[str, slice] = {
    "lohkomuoto": slice(8, 9),         # lohmuo (col 9)
    "section_y": slice(2, 5),          # lohy (col 3-5)
    "section_x": slice(5, 8),          # lohx (col 6-8)
    "test_area_number": slice(10, 12),  # koeala (col 11-12)
    "stand_number": slice(12, 13),     # kuvio (col 13)
    "tree_number": slice(15, 17),      # idnro (col 16-17)
    "species": slice(17, 19),          # puulaji (col 18-19)
    "diameter": slice(19, 22),         # d13 mm (col 20-22)
    "tree_category": slice(24, 25),    # puuluo (col 25)
    "latvuskerros": slice(26, 27),     # latker (col 27)
    "height": slice(61, 64),           # pituus dm (col 62-64)
    "living_branches_height": slice(58, 61),  # elalatva dm (col 59-61)
    "tuhon_ilmiasu": slice(83, 84),    # tuhilm (col 84)
    "d13_age": slice(91, 94),          # d13ika (col 92-94)
    "age_increase": slice(95, 97),     # ikalis (col 96-97)
    "total_age": slice(97, 100),       # kokika (col 98-100)
}


VMI10_STAND_INDICES: dict[str, slice] = {
    "lohkomuoto": slice(1, 2),          # lohmuo
    "section_y": slice(2, 5),           # lohy
    "section_x": slice(5, 8),           # lohx
    "test_area_number": slice(9, 11),   # koeala
    "stand_number": slice(12, 13),      # kuvio
    "row_type": slice(13, 14),          # tielaji (row type)
    "lat": slice(18, 25),               # pkoonim
    "lon": slice(25, 32),               # ikoonim
    "height_above_sea_level": slice(82, 87),  # korkeus (dm)
    "osuusrel": slice(36, 38),          # osuusrel
    "osuus7m": slice(40, 42),           # osuus7m
    "area_ha": slice(50, 59),           # eduala (ha, 5 decimals implied)
    "forestry_centre": slice(66, 68),  # metkes
    "municipality": slice(68, 71),      # kunta
    "degree_days": slice(87, 91),       # lamsum1
    "owner_group": slice(92, 93),       # omiryh (optional, may be unused)
    "date": slice(93, 99),              # pvm (ddmmyy)
    "land_category": slice(100, 101),           # maaluo 101
    "land_category_detail": slice(101, 102),    # maaluotar 102
    "paatyyppi": slice(104, 105),               # paatyy 105
    "kasvupaikkatunnus": slice(106, 107),       # kaspai 107
    "suotyy": slice(108, 110),
    "tkgtyy": slice(110, 111),
    "ojitus_tilanne": slice(126, 127),  # ojitil
    "ojitus_aika": slice(128, 130),     # ojiaik
    "ojitus_tarve": slice(130, 131),    # ojitar
    "tax_class": slice(132, 133),       # veroluo
    "tax_class_reduction": slice(133, 134),  # verotar
    "puuntuotannon_rajoitus": slice(135, 138),
    "puuntuotannon_rajoitus_tarkenne": slice(138, 139),
    "suojametsakoodi": slice(139, 140),  # suojamet
    "muut_arvot": slice(140, 141),       # muuarvo
    "naturaaluekoodi": slice(141, 142),  # natura
    "ahvenanmaan_markkinahakkuualue": slice(149, 150),
    "fra_class": slice(116, 117),               # fraluo 117
    "lat_measured": slice(183, 190),
    "lon_measured": slice(190, 197),
    "jakso1_asema": slice(200, 201),                 # jakasema (col 201)
    "kehitysluokka": slice(201, 202),                # kehluo
    "jakso1_syntytapa": slice(202, 203),             # syntapa (col 203)
    "jakso1_paapuulaji": slice(203, 205),            # paaplaji (col 204-205)
    "jakso1_paapuulaji_osuus": slice(205, 207),      # paaplajios (col 206-207)
    "jakso1_sivulaji1": slice(207, 209),             # sivplaji1 (col 208-209)
    "jakso1_sivulaji1_osuus": slice(209, 211),       # sivplaji1os (col 210-211)
    "jakso1_sivulaji2": slice(211, 213),             # sivplaji2 (col 212-213)
    "jakso1_sivulaji2_osuus": slice(213, 215),       # sivplaji2os (col 214-215)
    "jakso1_sivulaji3": slice(215, 217),             # sivplaji3 (col 216-217)
    "jakso1_sivulaji3_osuus": slice(217, 219),       # sivplaji3os (col 218-219)
    "jakso1_kokonaisrunkoluku1000": slice(226, 228),  # rlkok (col 227-228)
    "jakso1_keskilapimitta_cm": slice(231, 233),     # keskilpm (col 232-233)
    "jakso1_keskipituus_dm": slice(233, 236),        # keskipit (col 234-236)
    "jakso1_d13ika": slice(237, 240),                # d13ika (col 238-240)
    "jakso1_ikalisays": slice(241, 243),             # ikalis (col 242-243)
    "metsikon_ika": slice(255, 258),  # metika (cols 256-258)
    "hakkuu_tapa": slice(262, 263),
    "hakkuu_aika": slice(263, 264),
    "maanmuokkaus_aika": slice(269, 270),
    "viljely": slice(270, 271),
    "viljely_aika": slice(271, 272),
    "muu_toimenpide": slice(274, 275),
    "muu_toimenpide_aika": slice(275, 276),
    "hakkuuehdotus": slice(276, 277),  # hakehd1
    "ppa1": slice(284, 287),
    "ppa2": slice(288, 291),
    "ppa3": slice(292, 295),
    "ppa4": slice(296, 299),
    "ppa5": slice(300, 303),
    "basal_area": slice(305, 307),                  # kuvppa, kuvion ppa
    "jakso2_ppa": slice(307, 309),                   # j2ppa (cols 308-309)
    "jakso2_asema": slice(310, 311),                 # j2asema (col 311)
    "jakso2_syntytapa": slice(312, 313),             # j2syntapa (col 313)
    "jakso2_paapuulaji": slice(313, 315),            # j2paaplaji (col 314-315)
    "jakso2_paapuulaji_osuus": slice(315, 317),      # j2paaplajios (col 316-317)
    "jakso2_sivulaji1": slice(317, 319),             # j2sivplaji1 (col 318-319)
    "jakso2_sivulaji1_osuus": slice(319, 321),       # j2sivplaji1os (col 320-321)
    "jakso2_sivulaji2": slice(321, 323),             # j2sivplaji2 (col 322-323)
    "jakso2_sivulaji2_osuus": slice(323, 325),       # j2sivplaji2os (col 324-325)
    "jakso2_sivulaji3": slice(325, 327),             # j2sivplaji3 (col 326-327)
    "jakso2_sivulaji3_osuus": slice(327, 329),       # j2sivplaji3os (col 328-329)
    "jakso2_kokonaisrunkoluku1000": slice(336, 338),  # j2rlkok (col 337-338)
    "jakso2_keskilapimitta_cm": slice(341, 343),     # j2keskilpm (col 342-343)
    "jakso2_keskipituus_dm": slice(343, 346),        # j2keskipit (col 344-346)
    "jakso2_d13ika": slice(347, 350),                # j2d13ika (col 348-350)
    "jakso2_ikalisays": slice(351, 353),             # j2ikalis (col 352-353)
    "koealan_kasittelyluokka": slice(413, 416)
}

VMI10_TREE_INDICES: dict[str, slice] = {
    "lohkomuoto": slice(1, 2),          # lohmuo
    "section_y": slice(2, 5),           # lohy
    "section_x": slice(5, 8),           # lohx
    "test_area_number": slice(9, 11),   # koeala
    "stand_number": slice(12, 13),      # kuvio
    "row_type": slice(13, 14),          # tielaji
    "tree_type": slice(14, 15),         # puutyy
    "tree_number": slice(15, 17),       # idnro
    "species": slice(17, 19),           # puulaji
    "diameter": slice(19, 22),          # d13 (mm)
    "tree_category": slice(24, 25),     # puuluo
    "latvuskerros": slice(26, 27),      # latker
    "living_branches_height": slice(58, 61),  # elalatva (dm)
    "height": slice(61, 64),            # pituus (dm)
    "d13_age": slice(91, 94),           # d13ika (v)
    "age_increase": slice(95, 97),     # ikalis (v)
    "total_age": slice(97, 100),        # ika (v)
    "tuhon_ilmiasu": slice(83, 84),     # tuhilm
}


VMI11_STAND_INDICES: dict[str, slice] = {
    "inventointitunnus": slice(0, 1),  # (K=kerta, P=pysyvä)
    "lohkomuoto": slice(1, 2),
    "section_y": slice(2, 5),
    "section_x": slice(5, 8),
    "test_area_number": slice(9, 11),
    "ahvkeilaus": slice(11, 12),
    "stand_number": slice(12, 13),  # kuvio
    "row_type": slice(13, 14),      # tielaji
    "lat": slice(18, 25),  # pkoo
    "lon": slice(25, 32),  # ikoo
    "osuusrel": slice(36, 38),
    "osuus12x": slice(38, 40),
    "osuus7m": slice(40, 42),
    "area_ha": slice(50, 59),  # eduala
    "forestry_centre": slice(66, 68),  # metkes
    "municipality": slice(68, 71),    # kunta
    "height_above_sea_level": slice(82, 87),
    "degree_days": slice(87, 91),
    "owner_group": slice(92, 93),
    "date": slice(93, 99),
    "land_category": slice(100, 101),
    "land_category_detail": slice(101, 102),
    "fra_class": slice(102, 103),
    "paatyyppi": slice(104, 105),
    "kasvupaikkatunnus": slice(106, 107),
    "suotyy": slice(108, 110),
    "tkgtyy": slice(110, 111),
    "ojitus_tilanne": slice(126, 127),
    "ojitus_aika": slice(128, 130),
    "ojitus_tarve": slice(130, 131),
    "tax_class": slice(132, 133),
    "tax_class_reduction": slice(133, 134),
    "puuntuotannon_rajoitus": slice(135, 138),
    "puuntuotannon_rajoitus_tarkenne": slice(138, 139),
    "suojametsakoodi": slice(139, 140),
    "muut_arvot": slice(140, 141),
    "naturaaluekoodi": slice(141, 142),
    "ahvenanmaan_markkinahakkuualue": slice(149, 150),
    "lat_measured": slice(183, 190),
    "lon_measured": slice(190, 197),
    "kehitysluokka": slice(201, 202),
    "main_tree_species_dominant_storey": slice(202, 204),
    "alikehl": slice(205, 206),
    "ylikehl": slice(209, 210),
    "pohjapintaala": slice(228, 230),  # kuvppa
    "vallitsevanjakson_d13ika": slice(247, 250),
    "vallitsevanjakson_ikalisays": slice(250, 252),
    "hakkuu_tapa": slice(262, 263),
    "hakkuu_aika": slice(263, 264),
    "maanmuokkaus_aika": slice(269, 270),
    "viljely": slice(270, 271),
    "viljely_aika": slice(271, 272),
    "muu_toimenpide": slice(274, 275),
    "muu_toimenpide_aika": slice(275, 276),
    "hakkuuehdotus": slice(276, 277),  # hakehd1
    "koealan_kasittelyluokka": slice(304, 307),  # Koealan käsittelyluokka
}


VMI11_TREE_INDICES: dict[str, slice] = {
    "lohkomuoto": slice(1, 2),
    "section_y": slice(2, 5),
    "section_x": slice(5, 8),
    "test_area_number": slice(9, 11),
    "stand_number": slice(12, 13),
    "tree_type": slice(14, 15),
    "tree_number": slice(15, 17),
    "species": slice(17, 19),
    "diameter": slice(22, 24),  # d13cm
    "tree_category": slice(24, 25),
    "latvuskerros": slice(26, 27),
    "height": slice(61, 64),
    "living_branches_height": slice(58, 61),
    "measured_height": slice(61, 64),
    "tuhon_ilmiasu": slice(82, 84),
    "d13_age": slice(91, 94),
    "age_increase": slice(95, 97),
    "total_age": slice(97, 100),
}


VMI11_STRATUM_INDICES: dict[str, slice] = {
    "lohkomuoto": slice(1, 2),
    "section_y": slice(2, 5),
    "section_x": slice(5, 8),
    "test_area_number": slice(9, 11),
    "stand_number": slice(12, 13),
    "stratum_number": slice(15, 17),
    "stratum_rank": slice(19, 20),
    "species": slice(20, 22),
    "origin": slice(22, 23),
    "sapling_stems_per_ha": slice(24, 28),
    "stems_per_ha": slice(28, 33),
    "avg_diameter": slice(36, 38),
    "avg_height": slice(39, 42),
    "d13_age": slice(44, 47),
    "biological_age": slice(47, 49),  # ikalis
    "basal_area": slice(50, 52),
}


VMI12_STAND_INDICES: dict[str, slice] = {
    "lohkomuoto": slice(1, 2),
    "section_y": slice(2, 5),
    "section_x": slice(5, 8),
    "test_area_number": slice(9, 11),
    "stand_number": slice(12, 13),
    "row_type": slice(13, 14),
    "lat": slice(18, 25),
    "lon": slice(25, 32),
    "osuus9m": slice(36, 38),
    "osuus5m": slice(38, 40),
    "municipality": slice(45, 48),
    "county": slice(48, 50),
    "area_ha": slice(50, 59),
    "forestry_centre": slice(66, 68),
    "kitukunta": slice(68, 71),
    "height_above_sea_level": slice(82, 87),
    "degree_days": slice(87, 91),
    "owner_group": slice(92, 93),
    "date": slice(93, 99),
    "land_category": slice(100, 101),
    "land_category_detail": slice(101, 102),
    "fra_class": slice(102, 103),
    "paatyyppi": slice(104, 105),
    "kasvupaikkatunnus": slice(106, 107),
    "suotyy": slice(108, 110),
    "tkgtyy": slice(110, 111),
    "ojitus_tilanne": slice(126, 127),
    "ojitus_aika": slice(128, 130),
    "ojitus_tarve": slice(130, 131),
    "tax_class": slice(132, 133),
    "tax_class_reduction": slice(133, 134),
    "puuntuotannon_rajoitus": slice(135, 138),
    "puuntuotannon_rajoitus_tarkenne": slice(138, 139),
    "suojametsakoodi": slice(139, 140),
    "muut_arvot": slice(140, 141),
    "naturaaluekoodi": slice(141, 142),
    "ahvenanmaan_markkinahakkuualue": slice(149, 150),
    "lat_measured": slice(183, 190),
    "lon_measured": slice(190, 197),
    "kehitysluokka": slice(201, 202),
    "main_tree_species_dominant_storey": slice(202, 204),
    "alikehl": slice(205, 206),
    "ylikehl": slice(209, 210),
    "vallitsevanjakson_d13ika": slice(247, 250),
    "vallitsevanjakson_ikalisays": slice(250, 252),
    "hakkuu_tapa": slice(262, 263),
    "hakkuu_aika": slice(263, 264),
    "maanmuokkaus_aika": slice(269, 270),
    "viljely": slice(270, 271),
    "viljely_aika": slice(271, 272),
    "muu_toimenpide": slice(274, 275),
    "muu_toimenpide_aika": slice(275, 276),
    "hakkuuehdotus": slice(278, 289),
    "koealan_kasittelyluokka": slice(314, 317),
    "pohjapintaala": slice(228, 230)
}


VMI12_TREE_INDICES: dict[str, slice] = {
    "lohkomuoto": slice(1, 2),
    "section_y": slice(2, 5),
    "section_x": slice(5, 8),
    "test_area_number": slice(9, 11),
    "tree_type": slice(11, 12),
    "stand_number": slice(12, 13),
    "tree_number": slice(14, 17),
    "species": slice(17, 19),
    "diameter": slice(19, 22),
    "tree_category": slice(24, 25),
    "latvuskerros": slice(26, 27),
    "height": slice(36, 40),
    "origin": slice(54, 55),
    "living_branches_height": slice(58, 61),
    "measured_height": slice(61, 64),
    "tuhon_ilmiasu": slice(82, 84),
    "d13_age": slice(91, 94),
    "age_increase": slice(95, 97),
    "total_age": slice(97, 100),
}


VMI12_STRATUM_INDICES: dict[str, slice] = {
    "lohkomuoto": slice(1, 2),
    "section_y": slice(2, 5),
    "section_x": slice(5, 8),
    "test_area_number": slice(9, 11),
    "stand_number": slice(12, 13),
    "stratum_number": slice(15, 17),
    "stratum_rank": slice(19, 20),
    "species": slice(20, 22),
    "origin": slice(22, 23),
    "sapling_stems_per_ha": slice(24, 28),
    "stems_per_ha": slice(28, 33),
    "avg_diameter": slice(36, 38),
    "avg_height": slice(39, 42),
    "d13_age": slice(44, 47),
    "biological_age": slice(47, 49),
    "basal_area": slice(50, 52),
}


VMI13_STAND_INDICES: dict[str, int] = {
    "row_type": 0,
    "lohkomuoto": 2,
    "section_y": 3,
    "section_x": 4,
    "test_area_number": 5,
    "stand_number": 6,
    "lohkotarkenne": 7,
    "date": 9,
    "osuus9m": 14,
    "osuus4m": 15,
    "county": 17,
    "forestry_centre": 18,
    "municipality": 19,
    "kitukunta": 20,
    "owner_group": 24,
    "lat": 26,
    "lon": 27,
    "lat_measured": 28,
    "lon_measured": 29,
    "height_above_sea_level": 30,
    "degree_days": 31,
    "land_category": 40,
    "land_category_detail": 41,
    "fra_class": 46,
    "paatyyppi": 52,
    "kasvupaikkatunnus": 53,
    "suotyy": 55,
    "tkgtyy": 56,
    "ojitus_tilanne": 57,
    "ojitus_aika": 59,
    "ojitus_tarve": 60,
    "tax_class": 62,
    "tax_class_reduction": 63,
    "kehitysluokka": 70,
    "main_tree_species_dominant_storey": 71,
    "alikehl": 72,
    "ylikehl": 74,
    "pohjapintaala": 86,
    "vallitsevanjaksonika": 95,
    "hakkuu_tapa": 102,
    "hakkuu_aika": 103,
    "maanmuokkaus_aika": 109,
    "viljely": 110,
    "viljely_aika": 111,
    "muu_toimenpide": 113,
    "muu_toimenpide_aika": 114,
    "hakkuuehdotus": 115,
    "puuntuotannon_rajoitus": 125,
    "puuntuotannon_rajoitus_tarkenne": 126,
    "suojametsakoodi": 127,
    "muut_arvot": 128,
    "naturaaluekoodi": 129,
    "ahvenanmaan_markkinahakkuualue": 130,
    "koealan_kasittelyluokka": 131,
}


VMI13_TREE_INDICES: dict[str, int] = {
    "lohkomuoto": 2,
    "section_y": 3,
    "section_x": 4,
    "test_area_number": 5,
    "stand_number": 6,
    "tree_number": 7,
    "tree_type": 12,
    "species": 13,
    "diameter": 14,
    "tree_category": 16,
    "latvuskerros": 17,
    "height": 19,
    "origin": 25,
    "living_branches_height": 27,
    "measured_height": 28,
    "tuhon_ilmiasu": 38,
    "d13_age": 47,
    "age_increase": 49,
    "total_age": 50,
}


VMI13_STRATUM_INDICES: dict[str, int] = {
    "lohkomuoto": 2,
    "section_y": 3,
    "section_x": 4,
    "test_area_number": 5,
    "stand_number": 6,
    "stratum_number": 7,
    "stratum_rank": 11,
    "species": 12,
    "origin": 13,
    "sapling_stems_per_ha": 14,
    "stems_per_ha": 15,
    "avg_diameter": 16,
    "avg_height": 17,
    "d13_age": 19,
    "biological_age": 20,
    "basal_area": 23,
}


VMI12_COUNTY_AREAS = [
    341.144731908512, 333.997181334169, 0.0, 342.095800524934, 344.973457199735, 342.790305010893,
    337.97691292876, 341.680159256802, 344.538163001294, 334.106632294352, 388.636152954809,
    384.104671280277, 387.185442744553, 380.530710444382, 387.55872063968, 391.846213895394,
    455.901059564719, 773.027950310559, 451.83355704698, 791.65080, 10313.25275, 230.82912
]

# scots pine, norway spruce, silver birch, downy birch, aspen, alder
species_directly_mappable = ['1', '2', '3', '4', '5', '6']
# tervaleppä (??? alder?)
species_other_alder = ['7']
# other deciduous species
species_other_deciduous = ['8', '9', 'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'C1']
# other coniferous species
species_other_coniferous = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']

"""indeksit: metskeskusnumero, lohkomuoto. Määrittävät kuvion pinta-alan (palautuva arvo)"""
VMI10_COUNTY_AREAS = {
    0: {
        0: 132.35806,
    },
    1: {
        1: 352.41112,
        2: 314.61998,
    },
    2: {
        2: 310.52923,
    },
    3: {
        2: 317.64829,
    },
    4: {
        2: 324.71753,
    },
    5: {
        2: 314.38570,
    },
    6: {
        2: 306.43887,
    },
    7: {
        1: 349.94110,
    },
    8: {
        1: 347.11261,
        2: 347.11261,
    },
    9: {
        1: 349.54127,
    },
    10: {
        1: 348.70887,
    },
    11: {
        3: 415.77630,
    },
    12: {
        3: 417.33655,
        4: 868.86562,
    },
    13: {
        4: 850.52940,
    },
}

"""indeksit: metskeskusnumero, lohkomuoto. Määrittävät kuvion pinta-alan (palautuva arvo)"""
VMI11_COUNTY_AREAS = {
    0: {
        300: 100.39,
        400: 148.78,
    },
    1: {
        1: 350.45,
        2: 309.27,
    },
    2: {
        2: 314.34,
    },
    3: {
        2: 316.02,
    },
    4: {
        2: 312.38,
    },
    5: {
        2: 322.11,
    },
    6: {
        2: 310.01,
    },
    7: {
        1: 333.30,
    },
    8: {
        1: 335.54,
    },
    9: {
        1: 337.12,
    },
    10: {
        1: 336.79,
    },
    11: {
        3: 417.47,
    },
    12: {
        1: 339.89,
        3: 420.20,
        4: 874.84,
    },
    13: {
        4: 840.94,
        5: 10308.80,
    },
}
