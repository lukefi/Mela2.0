import sqlite3
from typing import Any
from lukefi.metsi.sim.runners import (
    Runner,
    default_runner)
from lukefi.metsi.sim.sim_configuration import SimConfiguration


def simulate_alternatives[T](control: dict[str, Any],
                             stands: list[T],
                             db: sqlite3.Connection,
                             runner: Runner[T] = default_runner):
    simconfig = SimConfiguration[T](**control)
    result = runner(stands, simconfig, db)
    return result
