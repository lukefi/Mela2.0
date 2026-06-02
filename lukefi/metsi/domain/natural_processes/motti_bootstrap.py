from typing import Any
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.natural_processes.motti_util import ensure_state

# TODO: why do we need this separate initialization when ensure_state is called in grow_motti_fn every time anyway?
def initialize_motti(stand: ForestStand, parameters: dict[str, Any]) -> None:
    _ = parameters
    if stand.motti_state is not None:
        return

    # TODO: Why is step=5 when just initializing?
    # TODO: In fact, why does ensure_state even take step as parameter?
    ensure_state(stand, step=5, sim_year=int(stand.year))
