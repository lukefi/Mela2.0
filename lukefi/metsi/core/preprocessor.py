from typing import Sequence

from lukefi.metsi.core.model import ComputationalUnit
from lukefi.metsi.core.sim_control import Preprocessing
from lukefi.metsi.core.runners import evaluate_sequence


def preprocess_units[T: ComputationalUnit](units: Sequence[T], control: Preprocessing[T]) -> Sequence[T]:
    return evaluate_sequence(units, *control.operations)
