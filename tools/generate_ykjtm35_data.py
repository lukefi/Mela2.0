import re
from pathlib import Path
import numpy as np


def parse_triangles(triangles_h: str) -> np.ndarray:
    """
    Parses triangles.h
    """
    rows = re.findall(r"\{(\s*\d+\s*),\s*(\d+)\s*,\s*(\d+)\s*\}", triangles_h)
    tri = np.array([[int(a), int(b), int(c)] for a, b, c in rows], dtype=np.int32)
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("Triangle parse failed")
    return tri


def parse_points(points_h: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses points.h
    """
    rows = re.findall(
        r"\{\s*(\d+)\s*,\s*([0-9\.\-]+)\s*,\s*([0-9\.\-]+)\s*,\s*([0-9\.\-]+)\s*,\s*([0-9\.\-]+)\s*,\s*([0-9\.\-]+)\s*,\s*([0-9\.\-]+)\s*\}",
        points_h
    )
    if not rows:
        raise ValueError("No points parsed")

    idx = np.array([int(r[0]) for r in rows], dtype=np.int32)
    coords = np.array([[float(r[1]), float(r[2]), float(r[3]), float(r[4])] for r in rows], dtype=np.float64)
    return idx, coords


def build_point_map(point_indices: np.ndarray) -> np.ndarray:
    """
    Builds point map
    """
    max_idx = int(point_indices.max())
    point_map = np.full(max_idx + 1, -1, dtype=np.int32)
    for i, ind in enumerate(point_indices):
        point_map[int(ind)] = i
    return point_map


def main() -> None:
    """
    Tool to convert triangles.h and points.h to ykjtm35_data.npz
    """

    src_dir = Path("../lukefi/metsi/forestry/c/source")
    triangles_path = src_dir / "triangles.h"
    points_path = src_dir / "points.h"

    triangles_h = triangles_path.read_text(encoding="utf-8", errors="replace")
    points_h = points_path.read_text(encoding="utf-8", errors="replace")

    triangles = parse_triangles(triangles_h)
    point_indices, ref_coords = parse_points(points_h)
    point_map = build_point_map(point_indices)

    out_dir = Path("../lukefi/metsi/data/preprocessing/")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ykjtm35_data.npz"

    np.savez_compressed(
        out_path,
        triangles=triangles,
        point_map=point_map,
        ref_coords=ref_coords,
    )
    print(f"Wrote {out_path} (triangles={triangles.shape}, ref_coords={ref_coords.shape})")


if __name__ == "__main__":
    main()
