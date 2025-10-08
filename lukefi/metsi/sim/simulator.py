import sqlite3
from typing import Any, Optional
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.sim.runners import (
    Runner,
    default_runner)
from lukefi.metsi.sim.sim_configuration import SimConfiguration


def simulate_alternatives[T: ComputationalUnit](control: dict[str, Any],
                                                stands: list[T],
                                                db: Optional[sqlite3.Connection] = None,
                                                runner: Runner[T] = default_runner):
    simconfig = SimConfiguration[T](**control)
    runner(stands, simconfig, db)
