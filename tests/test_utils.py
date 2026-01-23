"""
This module contains a collection of util functions and dummy payload functions for test cases
"""
import unittest
from typing import Any
from collections.abc import Callable

from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.data.enums.internal import LandUseCategory, SiteType, SoilPeatlandCategory, TreeSpecies
from lukefi.metsi.data.model import ForestStand, ReferenceTree
from lukefi.metsi.sim.simulation_payload import SimulationPayload


class ConverterTestSuite(unittest.TestCase):
    def run_with_test_assertions(self, assertions: list[tuple], fn: Callable):
        for case in assertions:
            result = fn(*case[0])
            self.assertEqual(case[1], result)


def raises(x: Any) -> None:
    raise Exception("Run failure")


def identity(x: Any) -> Any:
    return x


def none(x: Any) -> None:
    return None


def collect_results[T: ComputationalUnit](payloads: list[SimulationPayload[T]]) -> list[T]:
    return list(map(lambda payload: payload.computational_unit, payloads))


def prepare_growth_test_stand():
    stand = ForestStand(
        identifier="123",
        area=20.3,
        soil_peatland_category=SoilPeatlandCategory(1),
        site_type_category=SiteType(1),
        tax_class_reduction=1,
        land_use_category=LandUseCategory(1),
        geo_location=(
            6656996.0,
            310260.0,
            10.0,
            "EPSG:3067"),
        time=2025,
    )

    stand.reference_trees.create([
        {
            "identifier": "123-1",
            "species": int(TreeSpecies.PINE),
            "stems_per_ha": 123.0,
            "breast_height_diameter": 30.0,
            "height": 20.0,
            "biological_age": 55.0,
            "breast_height_age": 15.0,
            "sapling": False,
        },
        {
            "identifier": "123-2",
            "species": int(TreeSpecies.SPRUCE),
            "stems_per_ha": 123.0,
            "breast_height_diameter": 25.0,
            "height": 17.0,
            "biological_age": 37.0,
            "breast_height_age": 15.0,
            "sapling": False,
        },
        {
            "identifier": "123-3",
            "species": int(TreeSpecies.PINE),
            "stems_per_ha": 123.0,
            "breast_height_diameter": 0.0,
            "height": 0.3,
            "biological_age": 1.0,
            "breast_height_age": 0.0,
            "sapling": True,
        },
    ])
    return stand
