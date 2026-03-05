from typing import Any
from lukefi.metsi.data.model import ForestStand

# We reuse the predictor's ensure_state() so logic stays in one place
from lukefi.metsi.domain.natural_processes.grow_motti_dll import MottiDLLPredictor
from lukefi.metsi.sim.sim_configuration import SimConfiguration


def _iter_stands(unit: Any):
    """
    Yield ForestStand instances from possible CU shapes.
    - If unit is a ForestStand, yield it
    - If unit has attribute stands, yield ForestStand items inside
    - If unit has attribute forest_stands, same
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
    """
    if not config.transition.uses_motti:
        return

    data_dir = config.transition.parameters.get("data_dir")

    for stand in _iter_stands(unit):
        if getattr(stand, "motti_state", None) is not None:
            continue
        if not stand.has_trees():
            continue

        predictor = MottiDLLPredictor(stand, data_dir=data_dir)
        predictor.ensure_state(step=5, sim_year=int(stand.year))
