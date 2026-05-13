from typing import Any
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.motti_util import ensure_state


def initialize_motti(stand: ForestStand, parameters: dict[str, Any]) -> None:
    _ = parameters
    if stand.motti_state is not None:
        return

    ensure_state(stand, step=5, sim_year=int(stand.year))
