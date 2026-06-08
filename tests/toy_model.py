from sqlite3 import Connection
from typing import override
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.collected_data import CollectedData, OpTuple
from lukefi.metsi.sim.sim_configuration import Transition
from lukefi.metsi.sim.treatment import Treatment


class ToyModel(ComputationalUnit):
    value: int

    def __init__(self, identifier: str, value: int, time: int = 0) -> None:
        self.identifier = identifier
        self.value = value
        self.time = time
        self.start_time = time

    @override
    def output_to_db(self, db: Connection, node: str):
        cur = db.cursor()
        cur.execute(
            """--sql
                INSERT INTO toys
                VALUES (?, ?, ?, ?)
            """,
            (
                node,
                self.identifier,
                self.value,
                self.time
            )
        )

    @override
    def update_aggregates(self):
        pass

    @staticmethod
    def init_db_tables(db: Connection):
        cur = db.cursor()
        cur.execute(
        """--sql
            CREATE TABLE nodes(
                identifier TEXT,
                stand TEXT,
                done_treatment TEXT,
                treatment_params TEXT,
                tags TEXT,
                leaf INTEGER(1) DEFAULT(0),
                PRIMARY KEY(identifier, stand))
        """
        )
        cur.execute(
            """--sql
                CREATE TABLE toys (
                    node TEXT,
                    identifier TEXT,
                    value INTEGER,
                    time INTEGER,
                    PRIMARY KEY(node, identifier),
                    FOREIGN KEY(node, identifier) REFERENCES nodes(identifier, stand)
                )
            """
        )


class ToyTransition(Transition[ToyModel]):
    def __init__(self, max_step: int = 1, **parameters):
        super().__init__(toy_transition, max_step, **parameters)


def toy_transition(state: ToyModel, step: int = 1) -> OpTuple[ToyModel]:
    state.time += step
    return state, []


def toy_inc_fn(x: ToyModel, **operation_params) -> tuple[ToyModel, list[CollectedData]]:
    incrementation = operation_params.get("incrementation", 1)
    x.value += incrementation
    return x, []


def parametrized_treatment_fn(x: ToyModel, **kwargs) -> tuple[ToyModel, list[CollectedData]]:
    if kwargs.get('amplify') is True:
        x.value *= 1000
    return x, []


toy_inc = Treatment(toy_inc_fn, "toy_inc")
parametrized_treatment = Treatment(parametrized_treatment_fn, "parametrized_treatment")
