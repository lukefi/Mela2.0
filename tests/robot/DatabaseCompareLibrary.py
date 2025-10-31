import sqlite3

SELECT_NODES = "SELECT * FROM nodes ORDER BY stand"
SELECT_STANDS = "SELECT * FROM stands"
SELECT_STRATA = "SELECT * FROM strata"
SELECT_TREES = "SELECT * FROM trees"


def _compare(ref: str, res: str, select: str, numeric_tolerance = 0.0):
    ref_db = sqlite3.connect(ref)
    res_db = sqlite3.connect(res)
    ref_cur = ref_db.cursor()
    res_cur = res_db.cursor()
    ref_rows = ref_cur.execute(select)
    res_rows = res_cur.execute(select)
    for i, ref_row in enumerate(ref_rows):
        res_row = res_rows.fetchone()
        if res_row is None:
            raise AssertionError(f"Row {i} missing from test output: {ref_row} != {res_row}")
        for j, ref_col in enumerate(ref_row):
            if isinstance(ref_col, (int, float)):
                if not ref_col - numeric_tolerance <= res_row[j] <= ref_col + numeric_tolerance:
                    raise AssertionError(f"Difference in row {i}, column {j}: "
                                         f"{ref_col} != {res_row[j]}, tolerance {numeric_tolerance}")
            else:
                if ref_col != res_row[j]:
                    raise AssertionError(f"Difference in row {i}, column {j}: "
                                         f"{ref_col} != {res_row[j]}")


def node_tables_should_be_equal(ref: str, res: str):
    """Asserts that the `nodes` tables in the reference and result databases are identical, row by row.

    Args:
        ref (str): path to the reference database
        res (str): path to the result database
    """
    _compare(ref, res, SELECT_NODES)


def stand_tables_should_be_equal(ref: str, res: str, numeric_tolerance: float):
    """Asserts that the `stands` tables in the reference and result databases are identical, row by row.

    Args:
        ref (str): path to the reference database
        res (str): path to the result database
    """
    _compare(ref, res, SELECT_STANDS, numeric_tolerance)


def stratum_tables_should_be_equal(ref: str, res: str, numeric_tolerance: float):
    """Asserts that the `strata` tables in the reference and result databases are identical, row by row.

    Args:
        ref (str): path to the reference database
        res (str): path to the result database
    """
    _compare(ref, res, SELECT_STRATA, numeric_tolerance)


def tree_tables_should_be_equal(ref: str, res: str, numeric_tolerance: float):
    """Asserts that the `trees` tables in the reference and result databases are identical, row by row.

    Args:
        ref (str): path to the reference database
        res (str): path to the result database
    """
    _compare(ref, res, SELECT_TREES, numeric_tolerance)
