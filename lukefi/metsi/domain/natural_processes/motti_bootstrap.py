from typing import Any
from lukefi.metsi.data.model import ForestStand
import functools
# We reuse the predictor's ensure_state() so logic stays in one place
from lukefi.metsi.domain.natural_processes.grow_motti_dll import MottiDLLPredictor


def _extract_motti_data_dir(control: dict[str, Any]) -> str | None:
    """
    Tries a few common layouts so you don't have to freeze the control schema yet.
    """
    # flat key
    if "motti_data_dir" in control:
        return control.get("motti_data_dir")

    # nested
    m = control.get("motti") or control.get("natural_processes", {}).get("motti")
    if isinstance(m, dict):
        return m.get("data_dir") or m.get("motti_data_dir")

    return None


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


def _transition_uses_motti(control):
    t = control.get("transition")
    if t is None:
        return False

    # If Transition wrapper
    f = getattr(t, "transition_fn", None)
    if isinstance(f, functools.partial):
        base = f.func
        if getattr(base, "__name__", "") == "grow_motti_dll_fn":
            return True

    # fallback: callable transition itself
    if callable(t) and getattr(t, "__name__", "") == "grow_motti_dll_fn":
        return True

    return False


def ensure_motti_initialized(unit: Any, control: dict[str, Any]) -> None:
    """
    Initialize Motti state for all stands under this computational unit.

    Called once at simulation start from simulator.py.
    It is safe to call this multiple times; it is a no-op if stand.motti_state already exists.
    """
    if not _transition_uses_motti(control):
        return

    data_dir = _extract_motti_data_dir(control)

    for stand in _iter_stands(unit):
        # Only initialize if there are trees and we haven't already initialized.
        if getattr(stand, "motti_state", None) is not None:
            continue
        if not stand.has_trees():
            continue

        predictor = MottiDLLPredictor(stand, data_dir=data_dir)
        # step/year here don't matter much; they get overwritten each growth call anyway
        predictor.ensure_state(step=5, sim_year=int(stand.year))
