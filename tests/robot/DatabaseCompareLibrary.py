import sqlite3
import re

_SET_ITEM_RE = re.compile(r"'([^']*)'")


def _as_set_if_set_string(v):
    if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
        items = _SET_ITEM_RE.findall(v)
        if items:  # looks like a repr of a set of strings
            return set(items)
    return None


def _compare(ref: str, res: str, select: str, colum_names: str, numeric_tolerance=0.0):
    ref_db = sqlite3.connect(ref)
    res_db = sqlite3.connect(res)
    ref_db.row_factory = sqlite3.Row
    res_db.row_factory = sqlite3.Row
    ref_cur = ref_db.cursor()
    res_cur = res_db.cursor()
    ref_rows = ref_cur.execute(select)
    res_rows = res_cur.execute(select)
    ref_cur = ref_db.cursor()
    res_cur = res_db.cursor()

    for i, ref_row in enumerate(ref_rows):
        res_row = res_rows.fetchone()
        if res_row is None:
            raise AssertionError(f"Row {i} missing from test output: {ref_row} != {res_row}")
        keys = ref_row.keys()
        for key in keys:
            ref_col = ref_row[key]
            try:
                res_col = res_row[key]
            except IndexError as e:
                raise AssertionError(f"Column `{key}` missing from test output") from e

            if isinstance(ref_col, (int, float)):
                if not ref_col - numeric_tolerance <= res_col <= ref_col + numeric_tolerance:
                    raise AssertionError(f"Difference in row {i}, column `{key}`: "
                                         f"{ref_col} != {res_col}, tolerance {numeric_tolerance}")
            else:
                if ref_col != res_col:
                    ref_set = _as_set_if_set_string(ref_col)
                    res_set = _as_set_if_set_string(res_col)
                    if ref_set is not None and res_set is not None:
                        if ref_set != res_set:
                            raise AssertionError(f"Difference in row {i}, column `{key}`: {ref_col} != {res_col}")
                    else:
                        if ref_col != res_col:
                            raise AssertionError(f"Difference in row {i}, column `{key}`: {ref_col} != {res_col}")


def node_tables_should_be_equal(ref: str, res: str):
    """
    Asserts that the `nodes` tables in the reference and result databases are identical, row by row.

    Args:
        ref (str): path to the reference database
        res (str): path to the result database
    """
    NODE_COLUMNS = "PRAGMA table_info(nodes)"
    SELECT_NODES = "SELECT * FROM nodes ORDER BY unit"
    _compare(ref, res, SELECT_NODES, NODE_COLUMNS)


def stand_tables_should_be_equal(ref: str, res: str, numeric_tolerance: float):
    """
    Asserts that the `stands` tables in the reference and result databases are identical, row by row.

    Args:
        ref (str): path to the reference database
        res (str): path to the result database
    """
    STAND_COLUMNS = "PRAGMA table_info(stands)"
    SELECT_STANDS = "SELECT * FROM stands"
    _compare(ref, res, SELECT_STANDS, STAND_COLUMNS, numeric_tolerance)


def stratum_tables_should_be_equal(ref: str, res: str, numeric_tolerance: float):
    """
    Asserts that the `strata` tables in the reference and result databases are identical, row by row.

    Args:
        ref (str): path to the reference database
        res (str): path to the result database
    """
    STRATA_COLUMNS = "PRAGMA table_info(strata)"
    SELECT_STRATA = "SELECT * FROM strata"
    _compare(ref, res, SELECT_STRATA, STRATA_COLUMNS, numeric_tolerance)


def tree_tables_should_be_equal(ref: str, res: str, numeric_tolerance: float):
    """
    Asserts that the `trees` tables in the reference and result databases are identical, row by row.

    Args:
        ref (str): path to the reference database
        res (str): path to the result database
    """
    TREES_COLUMNS = "PRAGMA table_info(trees)"
    SELECT_TREES = "SELECT * FROM trees"
    _compare(ref, res, SELECT_TREES, TREES_COLUMNS, numeric_tolerance)


def removed_tree_tables_should_be_equal(ref: str, res: str, numeric_tolerance: float):
    """
    Asserts that the `removed_trees` tables in the reference and result databases are identical, row by row.

    Args:
        ref (str): path to the reference database
        res (str): path to the result database
    """
    REMOVED_TREES_COLUMNS = "PRAGMA table_info(removed_trees)"
    SELECT_REMOVED_TREES = "SELECT * FROM removed_trees"
    _compare(ref, res, SELECT_REMOVED_TREES, REMOVED_TREES_COLUMNS, numeric_tolerance)
