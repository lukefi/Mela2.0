import os
import sys
import sqlite3
from typing import Optional, Sequence, cast
from lukefi.metsi.app.metsi_enum import RunMode
from lukefi.metsi.app.preprocessor import (
    preprocess_stands,
)
from lukefi.metsi.app.app_io import parse_cli_arguments
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
from lukefi.metsi.sim.sim_control import MetsiControl
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.simulator import simulate_alternatives
from lukefi.metsi.app.console_logging import print_logline
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.sim.updating import update_units


def _preprocess(control: MetsiControl[ForestStand], stands: StandList) -> StandList:
    print_logline("Preprocessing...")
    preprocess_control = control.preprocessing
    if preprocess_control is not None:
        result = preprocess_stands(stands, preprocess_control)
        return result
    raise MetsiException("Declaration of preprocess control missing")


def _export_prepro(control: MetsiControl[ForestStand], data: StandList |
                   list[SimulationPayload[ForestStand]]) -> None:
    print_logline("Exporting preprocessing results...")
    if control.export_prepro is not None:
        return export_preprocessed(control.app_configuration.target_directory,
                                   control.export_prepro,
                                   data,
                                   base_name=control.app_configuration.preprocessing_output_file,
                                   app_configuration=control.app_configuration)
    print_logline("Declaration for 'export_prerocessed' not found from control.")
    print_logline("Skipping export of preprocessing results.")


def _update(control: MetsiControl[ForestStand], stands: StandList, db: sqlite3.Connection |
            None) -> tuple[StandList | list[SimulationPayload[ForestStand]], CollectableDataTypes | None]:
    updating_instructions = control.updating
    if updating_instructions is not None:
        print_logline(f"Updating stands to year {updating_instructions.target_time}...")
        return update_units(updating_instructions, stands, db)
    raise MetsiException("Declaration for 'updating' not found from control.")


def _simulate(control: MetsiControl[ForestStand],
              stands: StandList | list[SimulationPayload[ForestStand]],
              db: Optional[sqlite3.Connection],
              existing_data_types: CollectableDataTypes | None = None) -> None:
    simulation = control.simulation
    if simulation is not None:
        print_logline("Simulating alternatives...")
        return simulate_alternatives(simulation, stands, db, existing_data_types)
    raise MetsiException("Declaration for 'simulation' not found from control.")


def main() -> int:
    cli_arguments = parse_cli_arguments(sys.argv[1:])
    force_delete = bool(cli_arguments.pop("delete", False))

    control_file = "control.py" if cli_arguments["control_file"] is None else cli_arguments['control_file']
    try:
        control: MetsiControl[ForestStand] = read_control_module(control_file, cli_arguments)
        app_config = control.app_configuration
    except IOError:
        print(f"Application control file path '{control_file}' can not be read. Aborting....")
        return 1

    prepare_target_directory(app_config.target_directory)

    print_logline("Reading input...")
    should_continue = delete_existing_export_files(
        target_directory=app_config.target_directory,
        export_prepro=control.export_prepro,
        preprocessing_base_name=app_config.preprocessing_output_file,
        simulation_base_name=app_config.simulation_output_file,
        force_delete=force_delete,
    )

    if not should_continue:
        return 0

    db: sqlite3.Connection | None = None

    if RunMode.SIMULATE in app_config.run_modes or RunMode.UPDATE in app_config.run_modes:
        print_logline("Initializing output database")
        db_base = app_config.simulation_output_file or "simulation_results"
        db_name = db_base if str(db_base).lower().endswith(".db") else f"{db_base}.db"
        db = init_sqlite_database(f"{app_config.target_directory}/{db_name}")
        sqlite_decl = app_config.sqlite_decl
        create_database_tables(db, sqlite_decl=sqlite_decl)
        ForestStand.set_sqlite_decl(sqlite_decl)

    if app_config.run_modes[0] in [RunMode.PREPROCESS, RunMode.UPDATE, RunMode.SIMULATE]:
        input_data = read_stands_from_file(app_config, control.conversions or {})
    else:
        raise MetsiException("Can not determine input data for unknown run mode")

    prepare_target_directory(app_config.target_directory)

    cd_types_from_updating: CollectableDataTypes | None = None
    current: Sequence[ForestStand] | Sequence[SimulationPayload[ForestStand]] = input_data

    if RunMode.PREPROCESS in app_config.run_modes:
        current = _preprocess(control, cast(list[ForestStand], current))
    if RunMode.UPDATE in app_config.run_modes:
        current, cd_types_from_updating = _update(control, cast(list[ForestStand], current), db)
    if RunMode.EXPORT_PREPRO in app_config.run_modes:
        _export_prepro(control, current)
    if RunMode.SIMULATE in app_config.run_modes:
        _simulate(control, current, db, cd_types_from_updating)

    if db is not None:
        db.commit()
        db.close()

    _, dirs, files = next(os.walk(app_config.target_directory))
    if len(dirs) == 0 and len(files) == 0:
        os.rmdir(app_config.target_directory)

    print_logline("Exiting successfully")
    return 0


if __name__ == '__main__':
    main()
