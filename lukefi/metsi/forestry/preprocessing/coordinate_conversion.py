from enum import Enum
from typing import Optional
from numba import njit
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.forestry.preprocessing.data.points import ref_coords as _REF_COORDS, point_map as _POINT_MAP
from lukefi.metsi.forestry.preprocessing.data.triangles import triangles as _TRIANGLES


MAXTRIANGLE = 167000.0

_current_triangle_tm35_to_ykj = {"value": -1}


def _is_ykj(crs: Optional[str]) -> bool:
    return crs in CRS.EPSG_2393.value


def _is_erts(crs: Optional[str]) -> bool:
    return crs in CRS.EPSG_3067.value


@njit(cache=True)
def _check_triangle_nb(triangles, point_map, ref_coords, tri_index, x, y, input_is_ykj):
    coordoff = 0 if input_is_ykj else 2
    idxs = triangles[tri_index]

    # triangle vertices in the *input* CRS
    p0 = point_map[idxs[0]]
    p1 = point_map[idxs[1]]
    p2 = point_map[idxs[2]]

    ptsx0 = ref_coords[p0, coordoff]
    ptsy0 = ref_coords[p0, coordoff + 1]
    ptsx1 = ref_coords[p1, coordoff]
    ptsy1 = ref_coords[p1, coordoff + 1]
    ptsx2 = ref_coords[p2, coordoff]
    ptsy2 = ref_coords[p2, coordoff + 1]

    # early rejection like C
    t1 = ptsx0 - x
    t2 = ptsy0 - y
    if t1 * t1 + t2 * t2 > MAXTRIANGLE * MAXTRIANGLE:
        return False

    # cross products
    c1 = (ptsx1 - ptsx0) * (y - ptsy0) - (ptsy1 - ptsy0) * (x - ptsx0)
    c2 = (ptsx2 - ptsx1) * (y - ptsy1) - (ptsy2 - ptsy1) * (x - ptsx1)
    c3 = (ptsx0 - ptsx2) * (y - ptsy2) - (ptsy0 - ptsy2) * (x - ptsx2)

    o1 = (ptsx1 - ptsx0) * (ptsy2 - ptsy0) - (ptsy1 - ptsy0) * (ptsx2 - ptsx0) > 0.0
    o2 = (ptsx2 - ptsx1) * (ptsy0 - ptsy1) - (ptsy2 - ptsy1) * (ptsx0 - ptsx1) > 0.0
    o3 = (ptsx0 - ptsx2) * (ptsy1 - ptsy2) - (ptsy0 - ptsy2) * (ptsx1 - ptsx2) > 0.0

    it1 = (c1 == 0.0) or ((c1 > 0.0) == o1)
    it2 = (c2 == 0.0) or ((c2 > 0.0) == o2)
    it3 = (c3 == 0.0) or ((c3 > 0.0) == o3)
    return it1 and it2 and it3


@njit(cache=True)
def _find_triangle_nb(triangles, point_map, ref_coords, x, y, input_is_ykj, tri_hint):
    if tri_hint >= 0:
        if _check_triangle_nb(triangles, point_map, ref_coords, tri_hint, x, y, input_is_ykj):
            return tri_hint

    for i in range(triangles.shape[0]):
        if _check_triangle_nb(triangles, point_map, ref_coords, i, x, y, input_is_ykj):
            return i

    return -1


@njit(cache=True)
def _convert_using_triangle_nb(triangles, point_map, ref_coords, x, y, input_is_ykj, tri_index):
    in_off = 0 if input_is_ykj else 2
    out_off = 2 if input_is_ykj else 0

    idxs = triangles[tri_index]
    p0 = point_map[idxs[0]]
    p1 = point_map[idxs[1]]
    p2 = point_map[idxs[2]]

    # input CRS coords
    x0 = ref_coords[p0, in_off]
    y0 = ref_coords[p0, in_off + 1]
    x1 = ref_coords[p1, in_off]
    y1 = ref_coords[p1, in_off + 1]
    x2 = ref_coords[p2, in_off]
    y2 = ref_coords[p2, in_off + 1]

    # output CRS coords
    u0 = ref_coords[p0, out_off]
    v0 = ref_coords[p0, out_off + 1]
    u1 = ref_coords[p1, out_off]
    v1 = ref_coords[p1, out_off + 1]
    u2 = ref_coords[p2, out_off]
    v2 = ref_coords[p2, out_off + 1]

    c0x = y2 - y1
    c0y = x1 - x2
    c1x = y2 - y0
    c1y = x0 - x2
    c2x = y1 - y0
    c2y = x0 - x1

    w0 = ((x - x1) * c0x + (y - y1) * c0y) / ((x0 - x1) * c0x + (y0 - y1) * c0y)
    w1 = ((x - x0) * c1x + (y - y0) * c1y) / ((x1 - x0) * c1x + (y1 - y0) * c1y)
    w2 = ((x - x0) * c2x + (y - y0) * c2y) / ((x2 - x0) * c2x + (y2 - y0) * c2y)

    u = w0 * u0 + w1 * u1 + w2 * u2
    v = w0 * v0 + w1 * v1 + w2 * v2
    return u, v


class CRS(Enum):
    EPSG_3067 = ("EPSG:3067", "ERTS-TM35", "ETRS-TM35FIN")
    EPSG_2393 = ("EPSG:2393", "YKJ")

    @property
    def epsg(self) -> str:
        return self.value[0]

    @property
    def aliases(self) -> tuple[str, ...]:
        return self.value


def erts_tm35_to_ykj(u: float, v: float) -> tuple[float, float]:

    tri = _find_triangle_nb(
        _TRIANGLES, _POINT_MAP, _REF_COORDS,
        u, v,
        False,  # input_is_ykj
        _current_triangle_tm35_to_ykj["value"]
    )

    if tri < 0:
        raise MetsiException("Coordinate conversion failed: point not inside triangle mesh")

    _current_triangle_tm35_to_ykj["value"] = int(tri)

    x, y = _convert_using_triangle_nb(
        _TRIANGLES, _POINT_MAP, _REF_COORDS,
        u, v,
        False,
        tri
    )

    return float(x), float(y)


def convert_location_to_ykj(
    latitude: float,
    longitude: float,
    heigh_above_sea_level: Optional[float],
    crs: Optional[str],
) -> tuple[float, float, Optional[float], Optional[str]]:
    """Converts current coordinate system of the stand to match YKJ (EPSG:2393)."""

    if _is_ykj(crs):
        return (latitude, longitude, heigh_above_sea_level, crs)

    if _is_erts(crs):
        new_crs = CRS.EPSG_2393.epsg
        x, y = erts_tm35_to_ykj(latitude, longitude)
        return (x, y, heigh_above_sea_level, new_crs)

    raise MetsiException(
        f"Error while converting from {CRS.EPSG_3067.epsg} to {CRS.EPSG_2393.epsg}. "
        f"Check the source crs.\n We only support {CRS.EPSG_3067.epsg} as source crs at the moment."
    )


__all__ = ["convert_location_to_ykj", "CRS", "erts_tm35_to_ykj"]
