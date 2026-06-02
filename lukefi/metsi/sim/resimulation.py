import sqlite3
from typing import Any, Generator


def resimulate_schedules(control: dict[str, Any],
                         in_db: sqlite3.Connection,
                         out_db: sqlite3.Connection):
    _ = control
    _ = in_db
    _ = out_db

    # TODO: how to reconstruct initial state if and when original simulation db has incomplete attributes?
    #           - always complete output for initial state?
    #               - would lead to lots of mostly empty columns...
    #               - unless we add new tables specifically for the initial state?
    #           - require original source data or preprocessed csv?
    #               - how to deal with potential updating?
    # TODO: how to declare schedules/leaf nodes in control?
    # TODO: building simulation instructions (or equivalent) from declared leaf nodes/schedules
    # TODO: recreating dynamic parameters and other complex structures from original control file
    #           - possible quick hack - declare LUT in resim control?

    for schedule in _recreate_schedules(control, in_db):
        # resimulate schedule
        _ = schedule

def _recreate_schedules(control, in_db) -> Generator:
    _ = control
    _ = in_db
    yield from []
