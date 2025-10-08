import sqlite3
from typing import Optional
from copy import copy, deepcopy

from lukefi.metsi.app.utils import ConditionFailed
from lukefi.metsi.sim.finalizable import Finalizable
from lukefi.metsi.sim.simulation_payload import SimulationPayload, ProcessedTreatment
from lukefi.metsi.sim.state_tree import StateTree


def identity[T](x: T) -> T:
    return x


class EventTree[T]:
    """
    Event represents a computational operation in a tree of following event paths.
    """

    __slots__ = ('processed_treatment', 'branches')

    processed_treatment: ProcessedTreatment[T]
    branches: list["EventTree[T]"]

    def __init__(self, treatment: Optional[ProcessedTreatment[T]] = None):

        self.processed_treatment = treatment or identity
        self.branches = []

    def operation_chains(self) -> list[list[ProcessedTreatment[T]]]:
        """
        Recursively produce a list of lists of possible operation chains represented by this event tree in post-order
        traversal.
        """
        if len(self.branches) == 0:
            # Yes. A leaf node returns a single chain with a single operation.
            return [[self.processed_treatment]]
        result: list[list[ProcessedTreatment[T]]] = []
        for branch in self.branches:
            chains = branch.operation_chains()
            for chain in chains:
                result.append([self.processed_treatment] + chain)
        return result

    def evaluate(self,
                 payload: SimulationPayload[T],
                 node: list[int],
                 db: Optional[sqlite3.Connection] = None
                 ) -> list[SimulationPayload[T]]:
        """
        Recursive pre-order walkthrough of this event tree to evaluate its treatments with the given payload,
        copying it for branching. If given a root node, a StateTree is also constructed, containing all complete
        intermediate states in the simulation.

        :param payload: the simulation data payload (we don't care what it is here)
        :param state_tree: optional state tree node
        :return: list of result payloads from this EventTree or as concatenated from its branches
        """
        current = self.processed_treatment(payload)
        if db is not None:
            # state_tree.state = deepcopy(current.computational_unit)
            # state_tree.done_treatment = current.operation_history[-1][1] if len(current.operation_history) > 0 else None
            # state_tree.time_point = current.operation_history[-1][0] if len(current.operation_history) > 0 else None
            # state_tree.treatment_params = current.operation_history[-1][2] if
            # len(current.operation_history) > 0 else None
            node_str = "-".join(map(str, node))
            cur = db.cursor()
            cur.execute(
                """
                INSERT INTO nodes
                VALUES
                    (?, ?, ?, ?)
                """,
                (
                    node_str,
                    current.computational_unit.identifier,
                    str(current.operation_history[-1][1].__name__) if len(current.operation_history) > 0 else "do_nothing",
                    str(current.operation_history[-1][2]) if len(current.operation_history) > 0 else "{}"
                )
            )
            cur.execute(
                """
                INSERT INTO stands
                VALUES
                    (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    node_str,
                    current.computational_unit.identifier,
                    current.computational_unit.year,
                    current.computational_unit.management_unit_id,
                    current.computational_unit.stand_id,
                    current.computational_unit.area,
                    current.computational_unit.area_weight,
                    str(current.computational_unit.geo_location),
                    current.computational_unit.degree_days,
                    current.computational_unit.owner_category,
                    current.computational_unit.land_use_category,
                    current.computational_unit.soil_peatland_category,
                    current.computational_unit.site_type_category,
                    current.computational_unit.tax_class_reduction,
                    current.computational_unit.tax_class,
                    current.computational_unit.drainage_category,
                    current.computational_unit.drainage_feasibility,
                    current.computational_unit.drainage_year,
                    current.computational_unit.fertilization_year,
                    current.computational_unit.soil_surface_preparation_year,
                    current.computational_unit.natural_regeneration_feasibility,
                    current.computational_unit.regeneration_area_cleaning_year,
                    current.computational_unit.development_class,
                    current.computational_unit.artificial_regeneration_year,
                    current.computational_unit.young_stand_tending_year,
                    current.computational_unit.pruning_year,
                    current.computational_unit.cutting_year,
                    current.computational_unit.forestry_centre_id,
                    current.computational_unit.forest_management_category,
                    current.computational_unit.method_of_last_cutting,
                    current.computational_unit.municipality_id,
                    current.computational_unit.dominant_storey_age,
                    str(current.computational_unit.area_weight_factors),
                    current.computational_unit.fra_category,
                    current.computational_unit.land_use_category_detail,
                    current.computational_unit.auxiliary_stand,
                    current.computational_unit.sea_effect,
                    current.computational_unit.lake_effect,
                    current.computational_unit.basal_area))
            for i in range(current.computational_unit.reference_trees.size):
                cur.execute(
                    """
                    INSERT INTO trees
                    VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (node_str,
                     current.computational_unit.identifier,
                     current.computational_unit.reference_trees.identifier[i],
                     int(current.computational_unit.reference_trees.tree_number[i]),
                     int(current.computational_unit.reference_trees.species[i]),
                     current.computational_unit.reference_trees.breast_height_diameter[i],
                     current.computational_unit.reference_trees.height[i],
                     current.computational_unit.reference_trees.measured_height[i],
                     current.computational_unit.reference_trees.breast_height_age[i],
                     current.computational_unit.reference_trees.biological_age[i],
                     current.computational_unit.reference_trees.stems_per_ha[i],
                     int(current.computational_unit.reference_trees.origin[i]),
                     int(current.computational_unit.reference_trees.management_category[i]),
                     current.computational_unit.reference_trees.saw_log_volume_reduction_factor[i],
                     int(current.computational_unit.reference_trees.pruning_year[i]),
                     int(current.computational_unit.reference_trees.age_when_10cm_diameter_at_breast_height[i]),
                     f"({current.computational_unit.reference_trees.stand_origin_relative_position[i][0]}, {current.computational_unit.reference_trees.stand_origin_relative_position[i][1]}, {current.computational_unit.reference_trees.stand_origin_relative_position[i][2]})",
                     current.computational_unit.reference_trees.lowest_living_branch_height[i],
                     current.computational_unit.reference_trees.tree_category[i],
                     int(current.computational_unit.reference_trees.storey[i]),
                     bool(current.computational_unit.reference_trees.sapling[i]),
                     current.computational_unit.reference_trees.tree_type[i],
                     current.computational_unit.reference_trees.tuhon_ilmiasu[i]
                     )
                )
            for i in range(current.computational_unit.tree_strata.size):

                cur.execute(
                    """
                    INSERT INTO strata
                    VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (node_str,
                     current.computational_unit.identifier,
                     current.computational_unit.tree_strata.identifier[i],
                     int(current.computational_unit.tree_strata.species[i]),
                     current.computational_unit.tree_strata.mean_diameter[i],
                     current.computational_unit.tree_strata.mean_height[i],
                     current.computational_unit.tree_strata.breast_height_age[i],
                     current.computational_unit.tree_strata.biological_age[i],
                     current.computational_unit.tree_strata.stems_per_ha[i],
                     current.computational_unit.tree_strata.basal_area[i],
                     int(current.computational_unit.tree_strata.origin[i]),
                     int(current.computational_unit.tree_strata.management_category[i]),
                     current.computational_unit.tree_strata.saw_log_volume_reduction_factor[i],
                     int(current.computational_unit.tree_strata.cutting_year[i]),
                     int(current.computational_unit.tree_strata.age_when_10cm_diameter_at_breast_height[i]),
                     int(current.computational_unit.tree_strata.tree_number[i]),
                     f"({current.computational_unit.tree_strata.stand_origin_relative_position[i][0]}, {current.computational_unit.tree_strata.stand_origin_relative_position[i][1]}, {current.computational_unit.tree_strata.stand_origin_relative_position[i][2]})",
                     current.computational_unit.tree_strata.lowest_living_branch_height[i],
                     int(current.computational_unit.tree_strata.storey[i]),
                     current.computational_unit.tree_strata.sapling_stems_per_ha[i],
                     bool(current.computational_unit.tree_strata.sapling_stratum[i]),
                     int(current.computational_unit.tree_strata.number_of_generated_trees[i])
                     )
                )

        if isinstance(current.computational_unit, Finalizable):
            current.computational_unit.finalize()

        if len(self.branches) == 0:
            return [current]

        if len(self.branches) == 1:
            node_ = deepcopy(node)
            node_.append(0)
            return self.branches[0].evaluate(current, node_, db)

        results: list[SimulationPayload[T]] = []
        for i, branch in enumerate(self.branches):
            try:
                node_ = deepcopy(node)
                node_.append(i)
                evaluated_branch = branch.evaluate(copy(current), node_, db)
                results.extend(evaluated_branch)
            except (ConditionFailed, UserWarning):
                ...

        if len(results) == 0:
            raise UserWarning("Branch aborted with all children failing")

        return results

    def add_branch(self, et: 'EventTree[T]'):
        self.branches.append(et)
