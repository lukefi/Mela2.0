from dataclasses import dataclass
from typing import Generic, List, Optional, Sequence, TypeVar


T_co = TypeVar("T_co", covariant=True)


@dataclass
class ExportableContainer(Generic[T_co]):
    """ Output container for application results """
    export_objects: Sequence[T_co]
    additional_vars: Optional[List[str]]
