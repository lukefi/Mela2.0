"""
This module contains a collection of util functions and dummy payload functions for test cases
"""
from sqlite3 import Connection
import unittest
from typing import Any
from collections.abc import Callable

from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.data.enums.internal import LandUseCategory, SiteType, SoilPeatlandCategory, TreeSpecies
from lukefi.metsi.data.model import ForestStand, ReferenceTree
from lukefi.metsi.sim.collected_data import CollectedData
from lukefi.metsi.sim.simulation_payload import SimulationPayload


class ConverterTestSuite(unittest.TestCase):
    def run_with_test_assertions(self, assertions: list[tuple], fn: Callable):
        for case in assertions:
            result = fn(*case[0])
            self.assertEqual(case[1], result)


class DummyUnit(ComputationalUnit):
    x: int

    def __init__(self, x: int):
        self.x = x
    
    def output_to_db(self, db: Connection, node: str):
        pass

    def update_aggregates(self):
        pass


def raises(x: Any) -> None:
    raise Exception("Run failure")


def identity(x: Any) -> Any:
    return x


def none(x: Any) -> None:
    return None


def inc(x: DummyUnit,
        **operation_params) -> tuple[DummyUnit, list[CollectedData]]:
    incrementation = operation_params.get("incrementation", 1)
    x.x += incrementation
    return x, []


def parametrized_treatment(x: DummyUnit, **kwargs) -> tuple[DummyUnit, list[CollectedData]]:
    if kwargs.get('amplify') is True:
        x.x *= 1000
    return x, []


def collect_results(payloads: list[SimulationPayload]) -> list:
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
        reference_trees_pre_vec=[
            ReferenceTree(
                species=TreeSpecies.PINE,
                stems_per_ha=123,
                breast_height_diameter=30,
                height=20,
                biological_age=55,
                breast_height_age=15,
                sapling=False),
            ReferenceTree(
                species=TreeSpecies.SPRUCE,
                stems_per_ha=123,
                breast_height_diameter=25,
                height=17,
                biological_age=37,
                breast_height_age=15,
                sapling=False),
            ReferenceTree(
                species=TreeSpecies.PINE,
                stems_per_ha=123,
                breast_height_diameter=0,
                height=0.3,
                biological_age=1,
                breast_height_age=0,
                sapling=True)],
        year=2025)
    return stand
