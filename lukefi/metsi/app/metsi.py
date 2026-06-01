import os
import sys
import sqlite3
from typing import Any, Optional, cast
from lukefi.metsi.app.preprocessor import (
    preprocess_stands,
)
from lukefi.metsi.app.app_io import parse_cli_arguments, MetsiConfiguration, generate_application_configuration, RunMode
from lukefi.metsi.domain.forestry_types import StandList
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.app.export import export_preprocessed
from lukefi.metsi.app.file_io import (
    init_sqlite_database,
    prepare_target_directory,
    read_stands_from_file,
    delete_existing_export_files,
    read_control_module)
from lukefi.metsi.domain.utils.file_io import create_database_tables
from lukefi.metsi.sim.collected_data import CollectableDataTypes
from lukefi.metsi.sim.instructions import UpdatingInstructions
from lukefi.metsi.sim.resimulation import resimulate_schedules
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.simulator import simulate_alternatives
from lukefi.metsi.app.console_logging import print_logline
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.sim.updating import update_units


def _preprocess(control: dict[str, Any],
                stands: StandList) -> StandList:
    print_logline("Preprocessing...")
    result = preprocess_stands(stands, control)
    return result


def _export_prepro(config: MetsiConfiguration,
                   control: dict[str, Any],
                   data: StandList | list[SimulationPayload[ForestStand]]) -> None:
    print_logline("Exporting preprocessing results...")
    if control.get('export_prepro', None):
        export_preprocessed(config.target_directory, control['export_prepro'], data,
                            base_name=config.preprocessing_output_file,
                            app_configuration=control.get("app_configuration"))
    else:
        print_logline("Declaration for 'export_prerocessed' not found from control.")
        print_logline("Skipping export of preprocessing results.")


def _update(control: dict[str, Any],
            stands: StandList,
            db: sqlite3.Connection | None
            ) -> tuple[StandList | list[SimulationPayload[ForestStand]], CollectableDataTypes | None]:
    updating_instructions: UpdatingInstructions[ForestStand] | None = control.get("updating", None)
    if updating_instructions is not None:
        print_logline(f"Updating stands to year {updating_instructions.target_time}...")
        return update_units(updating_instructions, stands, db)

    raise MetsiException("Declaration for 'updating' not found from control.")


def _simulate(control: dict[str, Any],
              stands: StandList | list[SimulationPayload[ForestStand]],
              db: Optional[sqlite3.Connection],
              existing_data_types: CollectableDataTypes | None = None) -> None:
    print_logline("Simulating alternatives...")
    simulate_alternatives(control, stands, db, existing_data_types)


def _resimulate(control: dict[str, Any],
                in_db: sqlite3.Connection,
                out_db: sqlite3.Connection) -> None:
    print_logline("Resimulating schedules...")
    resimulate_schedules(control, in_db, out_db)


def main() -> int:
    cli_arguments = parse_cli_arguments(sys.argv[1:])
    force_delete = bool(cli_arguments.pop("delete", False))

    control_file = cli_arguments.get("control_file", MetsiConfiguration.control_file)

    try:
        control_structure = read_control_module(control_file)
    except IOError:
        print(f"Application control file path '{control_file}' can not be read. Aborting....")
        return 1

    app_config = generate_application_configuration({**cli_arguments, **control_structure['app_configuration']})

    prepare_target_directory(app_config.target_directory)

    print_logline("Reading input...")

    should_continue = delete_existing_export_files(
        target_directory=app_config.target_directory,
        export_prepro=control_structure.get("export_prepro"),
        preprocessing_base_name=app_config.preprocessing_output_file,
        simulation_base_name=app_config.simulation_output_file,
        force_delete=force_delete,
    )


    if not should_continue:
        return 0

    out_db: sqlite3.Connection | None = None

    if any(run_mode in app_config.run_modes for run_mode in [RunMode.UPDATE, RunMode.SIMULATE, RunMode.RESIMULATE]):
        print_logline("Initializing output database")
        db_base = app_config.simulation_output_file or "simulation_results"
        db_name = db_base if str(db_base).lower().endswith(".db") else f"{db_base}.db"
        out_db = init_sqlite_database(f"{app_config.target_directory}/{db_name}")
        sqlite_decl = control_structure['app_configuration'].get("sqlite_decl")
        create_database_tables(out_db, sqlite_decl=sqlite_decl)
        ForestStand.set_sqlite_decl(sqlite_decl)

    if app_config.run_modes[0] in [RunMode.PREPROCESS, RunMode.UPDATE, RunMode.SIMULATE]:
        input_data = read_stands_from_file(app_config, control_structure.get('conversions', {}))
    else:
        raise MetsiException("Can not determine input data for unknown run mode")

    prepare_target_directory(app_config.target_directory)

    cd_types_from_updating: CollectableDataTypes | None = None
    current: list[ForestStand] | list[SimulationPayload[ForestStand]] = input_data

    if RunMode.PREPROCESS in app_config.run_modes:
        current = _preprocess(control_structure, cast(list[ForestStand], current))
    if RunMode.UPDATE in app_config.run_modes:
        current, cd_types_from_updating = _update(control_structure, cast(list[ForestStand], current), out_db)
    if RunMode.EXPORT_PREPRO in app_config.run_modes:
        _export_prepro(app_config, control_structure, current)
    if RunMode.SIMULATE in app_config.run_modes:
        _simulate(control_structure, current, out_db, cd_types_from_updating)

    if RunMode.RESIMULATE in app_config.run_modes:
        assert out_db is not None
        in_db = sqlite3.Connection(app_config.input_path)
        _resimulate(control_structure, in_db, out_db)
        pass

    if out_db is not None:
        out_db.commit()
        out_db.close()

    _, dirs, files = next(os.walk(app_config.target_directory))
    if len(dirs) == 0 and len(files) == 0:
        os.rmdir(app_config.target_directory)

    print_logline("Exiting successfully")
    return 0


if __name__ == '__main__':
    main()
