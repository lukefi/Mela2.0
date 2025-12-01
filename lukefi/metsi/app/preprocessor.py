from lukefi.metsi.domain.forestry_types import StandList
from lukefi.metsi.sim.operations import simple_processable_chain
from lukefi.metsi.sim.runners import evaluate_sequence


def preprocess_stands(stands: StandList, simulation_declaration: dict) -> StandList:
    declared_operations = simulation_declaration.get('preprocessing_operations', {})
    preprocessing_params = simulation_declaration.get('preprocessing_params', {})
    preprocessing_funcs = simple_processable_chain(declared_operations, preprocessing_params)
    stands = evaluate_sequence(stands, *preprocessing_funcs)
    return stands


def slice_list_by_percentage[T](stands: list[T], percent: float) -> list[list[T]]:
    """Split `stands` into batches each containing approx `percent%` of the total."""
    total = len(stands)
    # at least one stand per batch
    batch_size = max(1, int(total * percent / 100.0))
    return [
        stands[i: i + batch_size]
        for i in range(0, total, batch_size)
    ]


def slice_list_by_size[T](stands: list[T], size: int) -> list[list[T]]:
    """Split `stands` into batches of up to `size` stands each."""
    total = len(stands)
    return [
        stands[i: i + size]
        for i in range(0, total, size)
    ]
