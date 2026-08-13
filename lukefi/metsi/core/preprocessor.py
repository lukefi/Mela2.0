from typing import Sequence

from lukefi.metsi.core.model import ComputationalUnit
from lukefi.metsi.core.sim_control import Preprocessing
from lukefi.metsi.core.operations import simple_processable_chain
from lukefi.metsi.core.runners import evaluate_sequence


def preprocess_units[T: ComputationalUnit](units: Sequence[T], control: Preprocessing[T]) -> Sequence[T]:
    declared_operations = control.operations
    preprocessing_params = control.params
    preprocessing_funcs = simple_processable_chain(declared_operations, preprocessing_params)
    return evaluate_sequence(units, *preprocessing_funcs)
