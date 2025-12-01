from typing import Any
import numpy as np
from lukefi.metsi.data.enums.internal import TreeSpecies, Storey


def is_nan(x: Any) -> bool:
    try:
        return bool(np.isnan(x))
    except TypeError:
        return False


def opt_float(x: Any) -> float | None:
    return None if is_nan(x) else float(x)


def opt_int(x: Any) -> int | None:
    # -1 is the default integer "missing" sentinel in VectorData
    return None if x is None or x == -1 else int(x)


def opt_species(code: Any) -> TreeSpecies | None:
    code_int = opt_int(code)
    return TreeSpecies(code_int) if code_int is not None else None


def opt_storey(code: Any) -> Storey | None:
    code_int = opt_int(code)
    return Storey(code_int) if code_int is not None else None
