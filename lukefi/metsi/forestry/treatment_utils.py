from typing import Mapping, Any
from lukefi.metsi.app.utils import MetsiException

def req(params: Mapping[str, Any], name: str) -> Any:
    try:
        return params[name]
    except KeyError as exc:
        raise MetsiException(
            f"Missing required regeneration parameter: '{name}'"
        ) from exc
