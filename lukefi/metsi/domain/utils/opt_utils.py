from typing import Any
from lukefi.metsi.data.enums.internal import TreeSpecies


def opt_int(x: Any) -> int | None:
    # -1 is the default integer "missing" sentinel in VectorData
    return None if x is None or x == -1 else int(x)


def opt_species(code: Any) -> TreeSpecies | None:
    code_int = opt_int(code)
    return TreeSpecies(code_int) if code_int is not None else None
