from copy import copy
import sqlite3
from pathlib import Path
from typing import Any, Optional
from lukefi.metsi.data.computational_unit import ComputationalUnit
from lukefi.metsi.domain.utils.file_io import output_node_to_db, update_leaf_node
from lukefi.metsi.sim.collected_data import init_collected_data_tables
from lukefi.metsi.sim.sim_configuration import SimConfiguration
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.app.utils import MetsiException


def simulate_alternatives[T: ComputationalUnit](control: dict[str, Any],
                                                units: list[T],
                                                db: Optional[sqlite3.Connection] = None):
    simconfig = SimConfiguration[T](control["simulation_instructions"], control["transition"], control["end_condition"])

    if db is not None:
        init_collected_data_tables(db, simconfig.collected_data)

    for unit in units:
        payload = SimulationPayload(unit)
        payload.computational_unit.update_aggregates()
        if db is not None:
            # Write initial state to database
            output_node_to_db(db, payload, [], {"initial"})
        _simulate_unit(payload, simconfig, db)


def _simulate_unit[T: ComputationalUnit](payload: SimulationPayload[T],
                                         config: SimConfiguration[T],
                                         db: Optional[sqlite3.Connection] = None) -> list[SimulationPayload[T]]:
    retval = []
    if not config.end_condition(payload):
        offset = 0
        all_instructions_failed = True
        for instruction in config.instructions:
            if all(condition(payload) for condition in instruction.conditions):
                all_instructions_failed = False
                for i, root in enumerate(instruction.unwrap()):
                    for new_branch in root.evaluate(copy(payload), db, i + offset):
                        new_branch.computational_unit, _ = config.transition(new_branch.computational_unit)
                        new_branch.computational_unit.update_aggregates()
                        retval.extend(_simulate_unit(new_branch, config, db))
                offset += 1
        if all_instructions_failed:
            # All instructions had failed conditions. Create one branch to carry on with transition.
            payload.computational_unit, _ = config.transition(payload.computational_unit)
            payload.computational_unit.update_aggregates()
            retval.extend(_simulate_unit(payload, config, db))
    else:
        # End condition met, update `leaf` column
        if db is not None:
            update_leaf_node(db, payload)
        retval = [payload]

    return retval


def delete_existing_export_files(
    target_directory: str,
    control: dict,
    preprocessing_base_name: str,
    force_yes: bool,
) -> bool:
    """
    Checks whether export_prepro output files already exist (csv/rst).
    If they exist:
      - if force_yes: delete and continue
      - else prompt user y/n
    Returns True if execution should continue, False if it should exit.
    """
    export_decl = control.get("export_prepro")
    if not export_decl:
        return True  # nothing will be written

    # Only consider formats declared in export_prepro
    formats = list(export_decl.keys())

    td = Path(target_directory)

    candidates: list[Path] = []
    for fmt in formats:
        candidates.append(td / f"{preprocessing_base_name}.{fmt}")
        if fmt == "rst":
            # rst_writer() may also write this file (avoid appending in repeated runs)
            candidates.append(td / "c-variables.par")

    existing = [p for p in candidates if p.is_file()]
    if not existing:
        return True

    if not force_yes:
        print("Output file(s) already exist:")
        for p in existing:
            print(f"  - {p}")
        answer = input("Do you want to delete them and continue? (y/n): ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborting (no files were deleted).")
            return False

    # delete
    for p in existing:
        try:
            p.unlink()
        except OSError as e:
            raise MetsiException(f"Unable to delete existing output file: {p}") from e

    return True
