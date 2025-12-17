from collections.abc import Generator
import sqlite3
from typing import Optional, TYPE_CHECKING
from copy import copy
from lukefi.metsi.sim.collected_data import CollectedData
from lukefi.metsi.app.utils import ConditionFailed
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.domain.utils.file_io import output_node_to_db
from lukefi.metsi.sim.finalizable import Finalizable
from lukefi.metsi.sim.simulation_payload import SimulationPayload
if TYPE_CHECKING:
    from lukefi.metsi.sim.collected_data import OpTuple
    from lukefi.metsi.sim.generators import ProcessedTreatment


def identity[T: ComputationalUnit](payload: SimulationPayload[T]) -> tuple[SimulationPayload[T], list[CollectedData]]:
    return payload, []


class EventTree[T: ComputationalUnit]:
    """
    Event represents a computational operation in a tree of following event paths.
    """

    __slots__ = ('processed_treatment', 'branches', 'tags')

    processed_treatment: "ProcessedTreatment[T]"
    branches: list["EventTree[T]"]
    tags: set[str]

    def __init__(self, treatment: Optional["ProcessedTreatment[T]"] = None, tags: Optional[set[str]] = None):

        self.processed_treatment = treatment or identity
        self.branches = []

        if tags is None:
            self.tags = set()
        else:
            self.tags = tags

    def evaluate(self,
                 payload: SimulationPayload[T],
                 db: Optional[sqlite3.Connection] = None,
                 node: Optional[int] = None,
                 ) -> Generator[SimulationPayload[T]]:
        """
        Recursive pre-order walkthrough of this event tree to evaluate its treatments with the given payload,
        copying it for branching. If a database connection is given, all simulated states and collected data is output
        to the database.

        :param payload: the simulation data payload (we don't care what it is here)
        :type payload: SimulationPayload[T]
        :param db: optional connection to an initialized database for output
        :type db: Optional[sqlite3.Connection]
        :param node: numeric identifier for the current simulator node, to be appended to the node_id of the payload
        :type node: Optional[int]
        :return: generator of result payloads from this EventTree or as concatenated from its branches
        :rtype: Generator[SimulationPayload[T], None, None]
        """
        if node is None:
            node = 0

        try:
            current, collected_data = self.processed_treatment(payload)
        except ConditionFailed:
            return

        current.node_id.append(node)

        if db is not None:
            output_node_to_db(db, current, collected_data, self.tags)

        if isinstance(current.computational_unit, Finalizable):
            current.computational_unit.finalize()

        if len(self.branches) == 0:
            yield current
            return

        if len(self.branches) == 1:
            yield from self.branches[0].evaluate(current, db)
            return

        for i, branch in enumerate(self.branches):
            yield from branch.evaluate(copy(current), db, i)

    def add_branch(self, et: 'EventTree[T]'):
        self.branches.append(et)
