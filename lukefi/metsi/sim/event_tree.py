import sqlite3
from typing import Optional
from copy import copy, deepcopy

from lukefi.metsi.app.utils import ConditionFailed
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.finalizable import Finalizable
from lukefi.metsi.sim.simulation_payload import SimulationPayload, ProcessedTreatment


def identity[T](x: T) -> T:
    return x


class EventTree[T: ComputationalUnit]:
    """
    Event represents a computational operation in a tree of following event paths.
    """

    __slots__ = ('processed_treatment', 'branches')

    processed_treatment: ProcessedTreatment[T]
    branches: list["EventTree[T]"]

    def __init__(self, treatment: Optional[ProcessedTreatment[T]] = None):

        self.processed_treatment = treatment or identity
        self.branches = []

    def evaluate(self,
                 payload: SimulationPayload[T],
                 node_identifier: Optional[list[int]] = None,
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
        if node_identifier is None:
            node_identifier = [0]
        if db is not None:
            node_str = "-".join(map(str, node_identifier))
            cur = db.cursor()
            cur.execute(
                """
                INSERT INTO nodes
                VALUES
                    (?, ?, ?, ?)
                """,
                (node_str,
                 current.computational_unit.identifier,
                 str(current.operation_history[-1][1].__name__) if len(current.operation_history) > 0 else "do_nothing",
                 str(current.operation_history[-1][2]) if len(current.operation_history) > 0 else "{}"))
            current.computational_unit.output_to_db(db, node_str)
            current.collected_data.output_to_db(db, node_str, current.computational_unit.identifier)

        if isinstance(current.computational_unit, Finalizable):
            current.computational_unit.finalize()

        if len(self.branches) == 0:
            return [current]

        if len(self.branches) == 1:
            node_identifier_ = deepcopy(node_identifier)
            node_identifier_.append(0)
            return self.branches[0].evaluate(current, node_identifier_, db)

        results: list[SimulationPayload[T]] = []
        for i, branch in enumerate(self.branches):
            try:
                node_identifier_ = deepcopy(node_identifier)
                node_identifier_.append(i)
                evaluated_branch = branch.evaluate(copy(current), node_identifier_, db)
                results.extend(evaluated_branch)
            except (ConditionFailed, UserWarning):
                ...

        if len(results) == 0:
            raise UserWarning("Branch aborted with all children failing")

        return results

    def add_branch(self, et: 'EventTree[T]'):
        self.branches.append(et)
