import os
import sys
import copy
import sqlite3
import traceback
from typing import Optional
from lukefi.metsi.app.preprocessor import (
    preprocess_stands,
    slice_list_by_percentage,
    slice_list_by_size
)
from lukefi.metsi.app.app_io import parse_cli_arguments, MetsiConfiguration, generate_application_configuration, RunMode
from lukefi.metsi.domain.forestry_types import StandList
from lukefi.metsi.app.export import export_preprocessed
from lukefi.metsi.app.file_io import (
    init_sqlite_database,
    prepare_target_directory,
    read_stands_from_file,
    delete_existing_export_files,
    read_control_module)
from lukefi.metsi.domain.utils.file_io import create_database_tables
from lukefi.metsi.sim.simulator import simulate_alternatives
from lukefi.metsi.app.console_logging import print_logline
from lukefi.metsi.app.utils import MetsiException


def _preprocess(config: MetsiConfiguration, control: dict, stands: StandList) -> StandList:
    _ = config
    print_logline("Preprocessing...")
    result = preprocess_stands(stands, control)
    return result


def _export_prepro(config: MetsiConfiguration, control: dict, data: StandList) -> None:
    print_logline("Exporting preprocessing results...")
    if control.get('export_prepro', None):
        export_preprocessed(config.target_directory, control['export_prepro'], data,
                            base_name=config.preprocessing_output_file)
    else:
        print_logline("Declaration for 'export_prerocessed' not found from control.")
        print_logline("Skipping export of preprocessing results.")


def _simulate(control: dict, stands: StandList, db: Optional[sqlite3.Connection]) -> None:
    print_logline("Simulating alternatives...")
    simulate_alternatives(control, stands, db)


def main() -> int:
    cli_arguments = parse_cli_arguments(sys.argv[1:])
    force_delete = bool(cli_arguments.pop("delete", False))

    control_file = \
        MetsiConfiguration.control_file if cli_arguments["control_file"] is None else cli_arguments['control_file']
    try:
        control_structure = read_control_module(control_file)
    except IOError:
        print(f"Application control file path '{control_file}' can not be read. Aborting....")
        return 1
    try:
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
        db: sqlite3.Connection | None = None
        if RunMode.SIMULATE in app_config.run_modes:
            print_logline("Initializing output database")
            db_base = app_config.simulation_output_file or "simulation_results"
            db_name = db_base if str(db_base).lower().endswith(".db") else f"{db_base}.db"
            db = init_sqlite_database(f"{app_config.target_directory}/{db_name}")
            create_database_tables(db)

        if app_config.run_modes[0] in [RunMode.PREPROCESS, RunMode.SIMULATE]:
            # 1) read full stand list
            full_stands = read_stands_from_file(app_config, control_structure.get('conversions', {}))

            # 2) split it if slice_* parameters are given
            pct = control_structure.get('slice_percentage')
            sz = control_structure.get('slice_size')
            if pct is not None:
                stand_sublists = slice_list_by_percentage(full_stands, pct)
            elif sz is not None:
                stand_sublists = slice_list_by_size(full_stands, sz)
            else:
                stand_sublists = [full_stands]

            input_data: list[StandList] = stand_sublists

        elif app_config.run_modes[0] in [RunMode.POSTPROCESS, RunMode.EXPORT]:
            raise MetsiException("Post-processing and export currently not implemented")

        else:
            raise MetsiException("Can not determine input data for unknown run mode")
    except Exception:  # pylint: disable=broad-exception-caught
        traceback.print_exc()
        print("Aborting run...")
        return 1

    # now run each slice in turn
    for _, stands in enumerate(input_data):
        # -- optional slice folder (disabled for now) --
        # slice_target = os.path.join(app_config.target_directory, f"slice_{slice_idx+1}")
        # prepare_target_directory(slice_target)

        # use original directory instead (to overwrite for now)
        prepare_target_directory(app_config.target_directory)

        # clone config so we don’t stomp on the original
        cfg = copy.copy(app_config)
        cfg.target_directory = app_config.target_directory

        # feed this sub‐list of stands through the normal run_modes
        current = stands
        if RunMode.PREPROCESS in cfg.run_modes:
            current = _preprocess(cfg, control_structure, current)
        if RunMode.EXPORT_PREPRO in cfg.run_modes:
            _export_prepro(cfg, control_structure, current)
        if RunMode.SIMULATE in cfg.run_modes:
            _simulate(control_structure, current, db)

    if db is not None:
        db.commit()
        db.close()

    _, dirs, files = next(os.walk(app_config.target_directory))
    if len(dirs) == 0 and len(files) == 0:
        os.rmdir(app_config.target_directory)

    print_logline("Exiting successfully")
    return 0


if __name__ == '__main__':
    sys.exit(main())
