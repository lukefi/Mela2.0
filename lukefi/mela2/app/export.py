import copy
from typing import Any, Callable, Optional

from lukefi.metsi.app.app_types import ExportableContainer
from lukefi.metsi.app.console_logging import print_logline
from lukefi.metsi.app.file_io import write_stands_to_file, determine_file_path
from lukefi.metsi.domain.forestry_types import StandList
from lukefi.metsi.sim.operations import simple_processable_chain
from lukefi.metsi.sim.runners import evaluate_sequence


def export_preprocessed(target_directory: str, decl: dict[str, Any], stands: StandList) -> None:
    output_formats = list(decl.keys())
    print_logline(f"Writing all preprocessed data to directory '{target_directory}'")
    for output_format in output_formats:
        operations: Optional[list[Callable[[StandList], StandList]]] = decl[output_format].get('operations', None)
        operation_params: Optional[dict[Callable, Any]] = decl[output_format].get('operation_params', None)
        additional_varnames: Optional[list[str]] = decl[output_format].get('additional_variables', None)
        file_name = f"preprocessing_result.{output_format}"
        filepaths = determine_file_path(target_directory, file_name)
        if operations is not None:
            operation_chain = simple_processable_chain(operations, operation_params or {})
            modified_stands = evaluate_sequence(copy.deepcopy(stands), *operation_chain)
            result = ExportableContainer(modified_stands, additional_varnames)
        else:
            result = ExportableContainer(stands, additional_varnames)
        print_logline(f"Writing preprocessed data to '{target_directory}\\{file_name}'")
        write_stands_to_file(result, filepaths, output_format)
