"""
This module contains a collection of util functions and dummy payload functions for test cases
"""
import unittest
from typing import Any, Optional
from collections.abc import Callable
import numpy as np

from lukefi.metsi.data.enums.internal import TreeSpecies
from lukefi.metsi.data.model import ForestStand, ReferenceTree
from lukefi.metsi.sim.collected_data import CollectedData
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


def inc(x: int, **operation_params) -> tuple[int, list[CollectedData]]:
    incrementation = operation_params.get("incrementation", 1)
    return x + incrementation, []


def parametrized_operation(x: SimulationPayload[int], **kwargs) -> SimulationPayload[int]:
    if kwargs.get('amplify') is True:
        x.computational_unit *= 1000
    return x


def collect_results(payloads: list[SimulationPayload]) -> list:
    return list(map(lambda payload: payload.computational_unit, payloads))


def prepare_growth_test_stand():
    stand = ForestStand(
        identifier="123",
        area=20.3,
        soil_peatland_category=1,
        site_type_category=1,
        tax_class_reduction=1,
        land_use_category=1,
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
