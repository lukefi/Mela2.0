from itertools import chain
from enum import Enum
from typing import Any, Optional
from collections.abc import Callable
import numpy as np
from lukefi.metsi.app.app_types import ExportableContainer
from lukefi.metsi.data.formats.util import parse_float
from lukefi.metsi.data.model import (
    ForestStand,
    ReferenceTree,
    TreeStratum,
    stand_as_internal_csv_row,
    stand_as_rst_row)
from lukefi.metsi.data.formats.rst_const import MSBInitialDataRecordConst as msb_meta
from lukefi.metsi.domain.forestry_types import StandList
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata


def rst_float(source: str | int | float) -> str:
    """
    Convert source to a float string with 6 decimals.
    """

    try:
        value = float(source)
        if np.isnan(value):
            value = 0.0
    except (TypeError, ValueError):
        value = 0.0

    return f"{value:.6f}"


def msb_metadata(stand: ForestStand) -> tuple[list[str], list[str], list[str]]:
    """
    Generate a triple with:
        MSB physical record metadata
        Initial data record stand metadata
        Initial data record tree set metadata
    """
    outputtable_id = parse_float(stand.identifier) or stand.stand_id or 0

    logical_record_length = sum([
        msb_meta.logical_record_metadata_length,
        msb_meta.stand_record_length,
        msb_meta.logical_subrecord_metadata_length,
        stand.reference_trees.size * msb_meta.tree_record_length
    ])
    physical_record_metadata = [
        rst_float(outputtable_id),  # UID
        str(sum([
            logical_record_length,
            msb_meta.logical_record_header_length
        ]))  # physical record length
    ]
    logical_record_metadata = [
        rst_float(msb_meta.logical_record_type),  # logical record type
        rst_float(logical_record_length),
        rst_float(msb_meta.stand_record_length)
    ]
    logical_subrecord_metadata = [
        rst_float(stand.reference_trees.size),
        rst_float(msb_meta.tree_record_length)
    ]
    return physical_record_metadata, logical_record_metadata, logical_subrecord_metadata


def c_var_metadata(uid: float | None, cvars_len: int) -> list[str]:
    total_length = 2 + cvars_len
    cvars_meta = map(rst_float, [uid or 0, total_length, 2, cvars_len])
    return list(cvars_meta)


def c_var_rst_row(stand: ForestStand, cvar_decl: list[str]) -> str:
    """ Content structure generation for a C-variable row """
    cvars_meta = c_var_metadata(parse_float(stand.identifier) or stand.stand_id, len(cvar_decl))
    cvars_row = " ".join(chain(
        cvars_meta,
        map(rst_float, stand.get_value_list(cvar_decl)
            )))
    return cvars_row


def rst_forest_stand_rows(stand: ForestStand, additional_vars: list[str]) -> list[str]:
    """Generate RST data file rows (with MSB metadata) for a single ForestStand"""
    result = []
    # Additional variables (C-variables) row
    if additional_vars:
        result.append(c_var_rst_row(stand, additional_vars))
    # Forest stand row
    msb_preliminary_records = msb_metadata(stand)
    result.append(" ".join(chain(
        msb_preliminary_records[0],
        msb_preliminary_records[1],
        map(rst_float, stand_as_rst_row(stand)),
        msb_preliminary_records[2]
    )))
    # Reference tree row(s)
    for i in range(stand.reference_trees.size):
        result.append(" ".join(map(rst_float, stand.reference_trees.as_rst_row(i))))
    return result


def csv_value(source: Any) -> str:
    if source in ("-1", "nan", "", None):
        return "None"
    return str(source)


def stand_to_csv_rows(stand: ForestStand, delimeter: str,
                      additional_vars: Optional[list[str]]) -> list[str]:
    """converts the :stand:, its reference trees and tree strata to csv rows."""
    result = []
    result.append(delimeter.join(map(csv_value, stand_as_internal_csv_row(stand, additional_vars))))

    for i in range(stand.reference_trees.size):
        result.append(delimeter.join(map(csv_value, stand.reference_trees.as_internal_csv_row(i))))

    for i in range(stand.tree_strata.size):
        result.append(delimeter.join(map(csv_value, stand.tree_strata.as_internal_csv_row(i))))

    return result


def stands_to_csv_content(container: ExportableContainer[ForestStand], delimeter: str) -> list[str]:
    stands = container.export_objects
    additional_vars = container.additional_vars
    result = []
    for stand in stands:
        result.extend(stand_to_csv_rows(stand, delimeter, additional_vars))
    return result


def _append_obj_to_attr_dict(attr_dict: dict[str, list[Any]], obj: Any) -> None:
    # obj is a dataclass-like instance produced by from_csv_row
    for k, v in obj.__dict__.items():
        if k == "stand":
            continue
        if isinstance(v, Enum):
            v = v.value
        attr_dict.setdefault(k, []).append(v)


def csv_content_to_stands(csv_content: list[list[str]]) -> StandList:
    stands: list[ForestStand] = []

    current_tree_attrs: dict[str, list[Any]] | None = None
    current_stratum_attrs: dict[str, list[Any]] | None = None

    def _finalize_current() -> None:
        if not stands:
            return
        stand = stands[-1]
        stand.reference_trees = ReferenceTrees().vectorize(current_tree_attrs or {})
        stand.tree_strata = TreeStrata().vectorize(current_stratum_attrs or {})

    for row in csv_content:
        row_type = row[0]

        if row_type == "stand":
            # finish previous stand before starting a new one
            _finalize_current()

            stands.append(ForestStand.from_csv_row(row))
            current_tree_attrs = {}
            current_stratum_attrs = {}

        elif row_type == "tree":
            t = ReferenceTree.from_csv_row(row)
            assert current_tree_attrs is not None, "Tree row encountered before first stand row"
            _append_obj_to_attr_dict(current_tree_attrs, t)

        elif row_type == "stratum":
            s = TreeStratum.from_csv_row(row)
            assert current_stratum_attrs is not None, "Stratum row encountered before first stand row"
            _append_obj_to_attr_dict(current_stratum_attrs, s)

    # finalize last stand
    _finalize_current()

    return stands


def outputtable_rows(container: ExportableContainer[ForestStand], formatter: Callable[[
                     ForestStand, list[str]], list[str]]) -> list[str]:
    stands = container.export_objects
    additional_vars = container.additional_vars or []
    result = []
    for stand in stands:
        result.extend(formatter(stand, additional_vars))
    return result


def stands_to_rst_content(container: ExportableContainer[ForestStand]) -> list[str]:
    """Generate RST file contents for the given list of ForestStand"""
    return outputtable_rows(container, rst_forest_stand_rows)


def mela_par_file_content(cvar_names: list[str]) -> list[str]:
    """ Par file content generalizes over all stands. Only single stand is needed """
    content = [f'#{vname.upper()}' for vname in cvar_names]
    content.insert(0, 'C_VARIABLES')
    return content
