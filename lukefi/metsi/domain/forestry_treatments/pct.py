from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.treatment import Treatment
from lukefi.metsi.domain.natural_processes.grow_motti_dll import (
    sync_ut_to_reference_trees,
    sync_yp_to_reference_trees,
    prune_reference_trees_not_in_motti,
)


def _normalize_species_array(value) -> list[int]:
    """
    Normalize caller-provided species-wise remaining counts into a 10-slot list.
    Slots 1..9 are species, slot 0 is unused.
    """
    if isinstance(value, dict):
        arr = [0] * 10
        for key, stems in value.items():
            idx = int(key)
            if not 1 <= idx <= 9:
                raise MetsiException(f"remaining_n_by_species index must be 1..9, got {idx}")
            arr[idx] = max(int(stems), 0)
        return arr

    vals = [int(x) for x in value]
    if len(vals) == 9:
        return [0] + [max(v, 0) for v in vals]
    if len(vals) == 10:
        out = [max(v, 0) for v in vals]
        out[0] = 0
        return out

    raise MetsiException(
        "remaining_n_by_species must be dict or list/tuple of length 9 or 10"
    )


def _scale_guidelines_to_total(guidelines: list[int], total: int) -> list[int]:
    """
    Backward compatibility for legacy 'remaining_n' scalar.
    Preserve the guideline species proportions, but scale the total.
    """
    if total <= 0:
        raise MetsiException("Parameter 'remaining_n' must be > 0")

    current = sum(max(int(x), 0) for x in guidelines[1:10])

    # If DLL guideline returns all zeros, fall back to putting the whole total
    # into the first species slot. This should be rare, but avoids crashing old flows.
    if current <= 0:
        out = [0] * 10
        out[1] = total
        return out

    raw = [0.0] * 10
    for i in range(1, 10):
        raw[i] = total * guidelines[i] / current

    out = [0] * 10
    for i in range(1, 10):
        out[i] = int(raw[i])

    # distribute rounding remainder to largest fractional parts
    remainder = total - sum(out[1:10])
    if remainder > 0:
        frac_order = sorted(
            range(1, 10),
            key=lambda i: (raw[i] - out[i]),
            reverse=True,
        )
        for i in frac_order[:remainder]:
            out[i] += 1

    return out


def _resolve_remaining_n(ms, operation_parameters) -> list[int]:
    """
    Preferred flow:
      1) ask Motti for guideline array
      2) optionally override or scale it
      3) pass the resulting species-wise array to Motti4PCT
    """
    guidelines = ms.dll.pct_guidelines_with_state(
        ms.yy,
        ms.yp,
        int(ms.ntrees),
        ms.buffers,
    )

    # New preferred parameter: explicit species-wise values
    if "remaining_n_by_species" in operation_parameters:
        return _normalize_species_array(operation_parameters["remaining_n_by_species"])

    # Legacy parameter: total scalar
    if "remaining_n" in operation_parameters:
        return _scale_guidelines_to_total(guidelines, int(operation_parameters["remaining_n"]))

    # Default: use Motti recommendation directly
    return guidelines


def pct_fn(input_: ForestStand, /, **operation_parameters) -> OpTuple[ForestStand]:
    """
    Motti-only sapling treatment.

    Preferred parameters:
      remaining_n_by_species:
          dict or list/tuple describing species-wise remaining stem counts
          for species slots 1..9

    Legacy parameter still supported:
      remaining_n: int
          total remaining stem count; guideline species proportions are preserved
    """
    stand = input_

    ms = getattr(stand, "motti_state", None)
    if ms is None or ms.buffers is None:
        raise MetsiException(
            "Motti PCT requested but stand has no initialized motti_state. "
            "Use Motti transition / bootstrap so state exists before this event."
        )

    remaining_n = _resolve_remaining_n(ms, operation_parameters)

    ms.ntrees = ms.dll.pct_with_state(
        ms.yy,
        ms.yp,
        int(ms.ntrees),
        ms.buffers,
        remaining_n=remaining_n,
    )

    # Keep Python-side vectors aligned with Motti after the treatment.
    sync_yp_to_reference_trees(stand)
    sync_ut_to_reference_trees(stand)
    prune_reference_trees_not_in_motti(stand)

    stand.young_stand_tending_year = stand.year

    return stand, []


pct = Treatment(pct_fn, "pct")
