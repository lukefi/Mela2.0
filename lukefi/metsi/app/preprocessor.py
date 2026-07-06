from lukefi.metsi.core.sim_control import Preprocessing
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.forestry_types import StandList
from lukefi.metsi.core.operations import simple_processable_chain
from lukefi.metsi.core.runners import evaluate_sequence


def preprocess_stands(stands: StandList, control: Preprocessing[ForestStand]) -> StandList:
    declared_operations = control.operations or []
    preprocessing_params = control.params or {}
    preprocessing_funcs = simple_processable_chain(declared_operations, preprocessing_params)
    stands = evaluate_sequence(stands, *preprocessing_funcs)
    return stands
