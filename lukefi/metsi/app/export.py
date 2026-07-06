import copy
import json
from typing import Any, Callable, Optional, Sequence

from lukefi.metsi.app.app_types import ExportableContainer
from lukefi.metsi.app.console_logging import print_logline
from lukefi.metsi.app.file_io import write_stands_to_file, determine_file_path
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.domain.forestry_types import StandList
from lukefi.metsi.sim.operations import simple_processable_chain
from lukefi.metsi.sim.runners import evaluate_sequence
from lukefi.metsi.app.metsi_control import AppConfiguration
from lukefi.metsi.sim.simulation_payload import SimulationPayload


def export_preprocessed(target_directory: str, decl: dict[str, Any],
                        units: StandList | Sequence[SimulationPayload[ForestStand]],
                        app_configuration: AppConfiguration,
                        base_name: str = "preprocessing_result",
                        ) -> None:
    output_formats = list(decl.keys())
    print_logline(f"Writing all preprocessed data to directory '{target_directory}'")

    stands: list[ForestStand] = []
    for unit in units:
        if isinstance(unit, SimulationPayload):
            stands.append(unit.computational_unit)
        else:
            stands.append(unit)

    for output_format in output_formats:
        operations: Optional[list[Callable[[StandList], StandList]]] = decl[output_format].get('operations', None)
        operation_params: Optional[dict[Callable, Any]] = decl[output_format].get('operation_params', None)
        additional_varnames: Optional[list[str]] = decl[output_format].get('additional_variables', None)
        file_name = f"{base_name}.{output_format}"
        filepaths = determine_file_path(target_directory, file_name)
        if operations is not None:
            operation_chain = simple_processable_chain(operations, operation_params or {})
            modified_stands = evaluate_sequence(copy.deepcopy(stands), *operation_chain)
            result = ExportableContainer(modified_stands, additional_varnames)
        else:
            result = ExportableContainer(stands, additional_varnames)
        print_logline(f"Writing preprocessed data to '{target_directory}\\{file_name}'")
        write_stands_to_file(result, filepaths, output_format)

        if output_format == "csv_exp":
            metadata_path = determine_file_path(target_directory, "metadata.json")
            payload = {
                "app_configuration": {
                    "input_path": app_configuration.input_path,
                    "target_directory": app_configuration.target_directory,
                    "slice_percentage": app_configuration.slice_percentage,
                    "slice_size": app_configuration.slice_size,
                    "state_format": app_configuration.state_format.value,
                    "run_modes": app_configuration.run_modes,
                    "preprocessing_output_file": app_configuration.preprocessing_output_file,
                    "simulation_output_file": app_configuration.simulation_output_file,
                    "sqlite_decl": app_configuration.sqlite_decl,
                    "measured_trees": app_configuration.measured_trees,
                    "strata": app_configuration.strata,
                    "strata_origin": app_configuration.strata_origin
                }
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
