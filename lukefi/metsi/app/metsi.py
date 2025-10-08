import os
import sys
import copy
import sqlite3
import traceback
from typing import Callable
from pathlib import Path
from lukefi.metsi.app.preprocessor import (
    preprocess_stands,
    slice_stands_by_percentage,
    slice_stands_by_size
)
from lukefi.metsi.app.app_io import parse_cli_arguments, MetsiConfiguration, generate_application_configuration, RunMode
from lukefi.metsi.domain.forestry_types import SimResults
from lukefi.metsi.domain.forestry_types import StandList
from lukefi.metsi.app.export import export_files, export_preprocessed
from lukefi.metsi.app.file_io import prepare_target_directory, read_stands_from_file, \
    read_full_simulation_result_dirtree, write_full_simulation_result_dirtree, read_control_module
from lukefi.metsi.app.post_processing import post_process_alternatives
from lukefi.metsi.domain.stand_runner import run_stands
from lukefi.metsi.sim.simulator import simulate_alternatives
from lukefi.metsi.app.console_logging import print_logline
from lukefi.metsi.app.utils import MetsiException


def preprocess(config: MetsiConfiguration, control: dict, stands: StandList) -> StandList:
    _ = config
    print_logline("Preprocessing...")
    result = preprocess_stands(stands, control)
    return result


def simulate(config: MetsiConfiguration, control: dict, stands: StandList) -> SimResults:
    print_logline("Simulating alternatives...")
    db_path = f"{config.target_directory}/simulation_results.db"
    try:
        os.remove(db_path)
    except OSError:
        pass
    db = sqlite3.connect(db_path)
    cur = db.cursor()
    cur.execute(
        """
        CREATE TABLE nodes(identifier, stand, done_treatment, treatment_params, PRIMARY KEY(identifier, stand))
        """
    )
    cur.execute(
        """
        CREATE TABLE stands(node, identifier, year, management_unit_id, stand_id, area, area_weight, geo_location,
                            degree_days, owner_category, land_use_category, soil_peatland_category,
                            site_type_category, tax_class_reduction, tax_class, drainage_category,
                            drainage_feasibility, drainage_year, fertilization_year,
                            soil_surface_preparation_year, natural_regeneration_feasibility,
                            regeneration_area_cleaning_year, development_class, artificial_regeneration_year,
                            young_stand_tending_year, pruning_year, cutting_year, forestry_centre_id,
                            forest_management_category, method_of_last_cutting, municipality_id,
                            dominant_storey_age, area_weight_factors, fra_category,land_use_category_detail,
                            auxiliary_stand, sea_effect, lake_effect, basal_area,
                            PRIMARY KEY(node, identifier),
                            FOREIGN KEY(node, identifier) REFERENCES nodes(identifier, stand))
        """
    )
    cur.execute(
        """
        CREATE TABLE trees(node, stand, identifier, tree_number, species, breast_height_diameter, height,
                           measured_height, breast_height_age, biological_age, stems_per_ha, origin,
                           management_category, saw_log_volume_reduction_factor, pruning_year,
                           age_when_10cm_diameter_at_breast_height, stand_origin_relative_position,
                           lowest_living_branch_height, tree_category, storey, sapling, tree_type, tuhon_ilmiasu,
                           PRIMARY KEY (node, identifier),
                           FOREIGN KEY (node, stand) REFERENCES nodes(identifier, stand))
        """
    )
    cur.execute(
        """
        CREATE TABLE strata(node, stand, identifier, species, mean_diameter, mean_height, breast_height_age,
                            biological_age, stems_per_ha, basal_area, origin, management_category,
                            saw_log_volume_reduction_factor, cutting_year, age_when_10cm_diameter_at_breast_height,
                            tree_number, stand_origin_relative_position, lowest_living_branch_height, storey,
                            sapling_stems_per_ha, sapling_stratum, number_of_generated_trees,
                            PRIMARY KEY (node, identifier),
                            FOREIGN KEY (node, stand) REFERENCES nodes(identifier, stand))
        """
    )

    result = simulate_alternatives(control, stands, db, run_stands)
    db.commit()
    db.close()
    if config.state_output_container is not None or config.derived_data_output_container is not None:
        print_logline(f"Writing simulation results to '{config.target_directory}'")
        write_full_simulation_result_dirtree(result, config)
    return result


def post_process(config: MetsiConfiguration, control: dict, data: SimResults) -> SimResults:
    print_logline("Post-processing alternatives...")
    result = post_process_alternatives(config, control['post_processing'], data)
    if config.state_output_container is not None or config.derived_data_output_container is not None:
        print_logline(f"Writing post-processing results to '{config.target_directory}'")
        write_full_simulation_result_dirtree(result, config)
    return result


def export(config: MetsiConfiguration, control: dict, data: SimResults) -> None:
    print_logline("Exporting simulation results...")
    if control['export']:
        export_files(config, control['export'], data)


def export_prepro(config: MetsiConfiguration, control: dict, data: StandList) -> StandList:
    print_logline("Exporting preprocessing results...")
    if control.get('export_prepro', None):
        export_preprocessed(config.target_directory, control['export_prepro'], data)
    else:
        print_logline("Declaration for 'export_prerocessed' not found from control.")
        print_logline("Skipping export of preprocessing results.")
    return data  # returned as is just for workflow reasons


def remove_existing_export_files(config: MetsiConfiguration, control: dict):
    """Remove known export and preprocessing output files from target_directory"""
    target_dir = Path(config.target_directory).resolve()
    safe_targets = set()

    # Collect export files from control["export"]
    if 'export' in control:
        for decl in control['export']:
            fmt = decl.get('format')
            if fmt == 'J':
                xda = decl.get('xda_filename', "data.xda")
                cda = decl.get('cda_filename', "data.cda")
                safe_targets.add(xda)
                safe_targets.add(cda)
            elif 'filename' in decl:
                safe_targets.add(decl['filename'])

    # Add preprocessing known output names
    if 'export_prepro' in control:
        for ext in control['export_prepro'].keys():
            safe_targets.add(f"preprocessing_result.{ext}")

    # Delete all collected files if they exist in the correct directory
    for filename in safe_targets:
        file_path = target_dir / filename
        try:
            if file_path.exists() and file_path.resolve().parent == target_dir:
                file_path.unlink()
        except (OSError, FileNotFoundError) as e:
            print_logline(f"Warning: Failed to delete file {file_path}: {e}")


mode_runners: dict[RunMode, Callable] = {
    RunMode.PREPROCESS: preprocess,
    RunMode.EXPORT_PREPRO: export_prepro,
    RunMode.SIMULATE: simulate,
    RunMode.POSTPROCESS: post_process,
    RunMode.EXPORT: export
}


def main() -> int:
    cli_arguments = parse_cli_arguments(sys.argv[1:])
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

        # deleting old target files
        remove_existing_export_files(app_config, control_structure)

        if app_config.run_modes[0] in [RunMode.PREPROCESS, RunMode.SIMULATE]:
            # 1) read full stand list
            full_stands = read_stands_from_file(app_config, control_structure.get('conversions', {}))

            # 2) split it if slice_* parameters are given
            pct = control_structure.get('slice_percentage')
            sz = control_structure.get('slice_size')
            if pct is not None:
                stand_sublists = slice_stands_by_percentage(full_stands, pct)
            elif sz is not None:
                stand_sublists = slice_stands_by_size(full_stands, sz)
            else:
                stand_sublists = [full_stands]

            input_data: list[StandList] | SimResults = stand_sublists

        elif app_config.run_modes[0] in [RunMode.POSTPROCESS, RunMode.EXPORT]:
            input_data = read_full_simulation_result_dirtree(app_config.input_path)
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
        for mode in cfg.run_modes:
            runner = mode_runners[mode]
            current = runner(cfg, control_structure, current)

    _, dirs, files = next(os.walk(app_config.target_directory))
    if len(dirs) == 0 and len(files) == 0:
        os.rmdir(app_config.target_directory)

    print_logline("Exiting successfully")
    return 0


if __name__ == '__main__':
    sys.exit(main())
