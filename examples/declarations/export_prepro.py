from typing import Any

from lukefi.metsi.domain.exp_ops import classify_values_to, prepare_rst_output


default: dict[str, Any] = {}  # Empty dict declares a default output content

mela_decl = {
    'operations': [prepare_rst_output, classify_values_to],
    'operation_params': {
        classify_values_to: [
            {'format': 'rst'}
        ]
    }
}


mela = {
    'rst': mela_decl,
}

mela_and_default_csv = {
    'rst': mela_decl,
    'csv': default,
}

default_csv = {
    'csv': default
}

default_csv_exp = {
    'csv_exp': default,
}

mela_csv_legacy_and_exp = {
    'rst': mela_decl,
    'csv': default,
    'csv_exp': default,
}
__all__ = [
    'mela',
    'default_csv',
    'default_csv_exp',
    'mela_and_default_csv',
    'mela_csv_legacy_and_exp',
]
