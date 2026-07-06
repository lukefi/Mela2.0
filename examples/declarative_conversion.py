""" An control file example to demonstrate to usage on declarative variables and exportin of the suchs variables """

from random import random
from examples.declarations.export_prepro import mela
from lukefi.metsi.app.metsi_enum import RunMode, StateFormat
from lukefi.metsi.core.sim_control import Preprocessing
from lukefi.metsi.data.formats.declarative_conversion import Conversion
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.app.metsi_control import AppConfiguration, MetsiControl


def sum3(x, y, z) -> float:
    return float(x) + float(y) + float(z)


control_structure = MetsiControl[ForestStand](
    app_configuration=AppConfiguration(
        state_format=StateFormat.VMI13,
        run_modes=[RunMode.PREPROCESS, RunMode.EXPORT_PREPRO]
    ),
    # Examples of declarative conversions
    conversions={
        'vmi13': {
            # common conversions
            'VAR0': Conversion(lambda: 123456789),
            'VAR1': Conversion(lambda x: int(x) * 2, indices=("row_type",)),
            'VAR2': Conversion(lambda x: x, indices=("lohkomuoto",)),
            'VAR3': Conversion(sum3, indices=("lohkomuoto", "section_y", "section_x")),
            'VAR4': Conversion(lambda x, y: pow(int(x), int(y)), indices=("lohkomuoto", "test_area_number")),
            'VAR5': Conversion(lambda x, y: pow(float(x), float(y)), indices=("section_y", "test_area_number")),
            'VAR_RANDOM': Conversion(random),
            'VAR_KISSA': Conversion(lambda: "Kissa123"),
            'VAR8': Conversion(lambda x: str(x) if not isinstance(x, str) else x, indices=("section_y",)),
            # conversions based on object type spesifications
            'VAR9': Conversion(lambda x, obj: int(x) * obj.area, indices=("row_type",), object_type=ForestStand),
            'VAR10': Conversion(lambda x, obj: int(x) * obj.VAR1, indices=("row_type",), object_type=ForestStand),
        }
    },
    preprocessing=Preprocessing(
        operations=[],
        params={}
    ),
    export_prepro=mela
)


__all__ = ['control_structure']
