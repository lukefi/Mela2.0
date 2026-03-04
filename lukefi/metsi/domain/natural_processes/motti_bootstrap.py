from typing import Any
from lukefi.metsi.data.model import ForestStand

# We reuse the predictor's ensure_state() so logic stays in one place
from lukefi.metsi.domain.natural_processes.grow_motti_dll import MottiDLLPredictor
from lukefi.metsi.sim.sim_configuration import SimConfiguration


def _iter_stands(unit: Any):
    """
    Yield ForestStand instances from possible 'computational unit' shapes.
    - If unit is a ForestStand, yield it
    - If unit has attribute 'stands' (list-like), yield ForestStand items inside
    - If unit has attribute 'forest_stands' (list-like), same
    Otherwise: do nothing.
    """
    if isinstance(unit, ForestStand):
        yield unit
        return

    for attr in ("stands", "forest_stands"):
        if hasattr(unit, attr):
            items = getattr(unit, attr)
            try:
                for s in items:
                    if isinstance(s, ForestStand):
                        yield s
            except TypeError:
                pass


def ensure_motti_initialized(unit: Any, config: SimConfiguration) -> None:
    """
    Initialize Motti state for all stands under this computational unit.

    Called once at simulation start from simulator.py.
    It is safe to call this multiple times; it is a no-op if stand.motti_state already exists.
    """
    if not config.transition.uses_motti:
        return

    data_dir = config.transition.parameters.get("data_dir")

    for stand in _iter_stands(unit):
        # Only initialize if there are trees and we haven't already initialized.
        if getattr(stand, "motti_state", None) is not None:
            continue
        if not stand.has_trees():
            continue

        predictor = MottiDLLPredictor(stand, data_dir=data_dir)
        # step/year here don't matter much; they get overwritten each growth call anyway
        predictor.ensure_state(step=5, sim_year=int(stand.year))
