from copy import copy
from typing import Any, Optional, overload
import numpy as np
import numpy.typing as npt

from lukefi.metsi.app.utils import MetsiException

DTYPES_TREE: dict[str, npt.DTypeLike] = {
    "identifier": np.dtype("U20"),
    "tree_number": np.int32,
    "species": np.int32,
    "breast_height_diameter": np.float64,
    "height": np.float64,
    "measured_height": np.float64,
    "breast_height_age": np.float64,
    "biological_age": np.float64,
    "stems_per_ha": np.float64,
    "origin": np.int32,
    "management_category": np.int32,
    "saw_log_volume_reduction_factor": np.float64,
    "pruning_year": np.int16,
    "age_when_10cm_diameter_at_breast_height": np.int16,
    "stand_origin_relative_position": np.dtype((np.float64, (3,))),
    "lowest_living_branch_height": np.float64,
    "tree_category": np.str_,
    "storey": np.int32,
    "sapling": np.bool_,
    "tree_type": np.dtype("U20"),
    "tuhon_ilmiasu": np.dtype("U20"),
}

DTYPES_STRATA: dict[str, npt.DTypeLike] = {
    "identifier": np.dtype("U20"),
    "species": np.int32,
    "mean_diameter": np.float64,
    "mean_height": np.float64,
    "breast_height_age": np.float64,
    "biological_age": np.float64,
    "stems_per_ha": np.float64,
    "basal_area": np.float64,
    "origin": np.int32,
    "management_category": np.int32,
    "saw_log_volume_reduction_factor": np.float64,
    "cutting_year": np.int32,
    "age_when_10cm_diameter_at_breast_height": np.int16,
    "tree_number": np.int32,
    "stand_origin_relative_position": np.dtype((np.float64, (3,))),
    "lowest_living_branch_height": np.float64,
    "storey": np.int32,
    "sapling_stems_per_ha": np.float64,
    "sapling_stratum": np.bool_,
    "number_of_generated_trees": np.int32
}


class VectorData():
    """
    Base class for generic SoA data.
    """
    dtypes: dict[str, npt.DTypeLike]
    size: int

    def __init__(self, dtypes: dict[str, npt.DTypeLike]):
        self.dtypes = dtypes
        self.vectorize({})

    def __len__(self):
        return self.size

    def __getitem__(self, name: str) -> npt.NDArray:
        return getattr(self, name)

    def vectorize(self, attr_dict: dict[str, list[Any]]):
        self.set_size(attr_dict)
        for attribute_name, data_type in self.dtypes.items():
            setattr(
                self,
                attribute_name,
                np.array(
                    self.defaultify(
                        attr_dict.get(
                            attribute_name,
                            [None] *
                            self.size),
                        data_type),
                    data_type))
            if not self.is_contiguous(attribute_name):
                raise MetsiException("Vectorized data is not contiguous")
        return self

    def is_contiguous(self, name: str):
        arr: npt.NDArray = getattr(self, name)
        return bool(arr.flags['CONTIGUOUS']) and bool(arr.flags['C_CONTIGUOUS'])

    def set_size(self, attr_dict: dict[str, list[Any]]):
        size = len(attr_dict.get('identifier', []))
        setattr(self, 'size', size)

    def defaultify(self, values: list, dtype: npt.DTypeLike) -> list:
        return [self.to_default(v, dtype) for v in values]

    def to_default(self, value: Optional[Any], field_type: npt.DTypeLike) -> Any:
        """ Replace None with appropriate defaults based on field type. """
        int_default = -1
        str_default = ""
        float_default = np.nan
        bool_default = False
        tuple_default = (np.nan, np.nan, np.nan)
        object_default = None
        retval: Any

        if value is None:
            if np.issubdtype(field_type, np.integer):
                retval = int_default
            elif np.issubdtype(field_type, np.floating):
                retval = float_default
            elif np.issubdtype(field_type, np.str_):
                retval = str_default
            elif np.issubdtype(field_type, np.bool_):
                retval = bool_default
            elif np.issubdtype(field_type, np.void):
                retval = tuple_default
            else:
                retval = object_default
            return retval
        return value

    @overload
    def create(self, new: dict[str, Any], index: int | None = None):
        ...

    @overload
    def create(self, new: list[dict[str, Any]], index: list[int] | None = None):
        ...

    def create(self, new: dict[str, Any] | list[dict[str, Any]], index: int | list[int] | None = None):
        """
        Creates a new row of data for all arrays contained in the data type. Default values are used for unspecified
        columns.

        Args:
            new (dict[str, Any] | list[dict[str, Any]]): A dictionary, or list of dictionaries, mapping attribute names
                                                         to new values.
            index (int | list[int] | None, optional): Index or list of indices where to insert the new rows.
                                                      If not given, values are appended to the ends of the arrays.
                                                      Defaults to None.
        """
        def _row_block_like(col: np.ndarray, val, dtype):
            tail = col.shape[1:]
            if tail:
                arr = np.asarray(val, dtype=dtype).reshape(tail)   # (p,), (p,q) ...
                return arr.reshape((1,)+tail)                      # (1,p) or (1,p,q)

            return np.asarray([val], dtype=dtype)              # (1,)

        def _many_block_like(col: np.ndarray, vals_list, dtype):
            tail = col.shape[1:]
            if tail:
                stacked = np.stack([np.asarray(v, dtype=dtype).reshape(tail) for v in vals_list], axis=0)  # (m,*tail)
                return stacked

            return np.asarray(vals_list, dtype=dtype).reshape(-1, *tail)  # (m,)

        def _concat(col: np.ndarray, block: np.ndarray, at: int | None):
            if at is None:
                return np.concatenate([col, block], axis=0)
            return np.concatenate([col[:at], block, col[at:]], axis=0)

        if isinstance(new, list):
            # Prepare per-column blocks
            blocks = {}
            for key, dtype in self.dtypes.items():
                col = getattr(self, key)
                vals_list = [self.to_default(item.get(key), dtype) for item in new]
                blocks[key] = _many_block_like(col, vals_list, dtype)

            # Apply append / block-insert / per-row inserts
            if index is None or isinstance(index, int):
                for key in self.dtypes.keys():
                    col = getattr(self, key)
                    setattr(self, key, _concat(col, blocks[key], index if isinstance(index, int) else None))
            else:
                # index is list[int], same length as new
                idx_list = list(index)
                if len(idx_list) != len(new):
                    raise ValueError("Length of 'index' must match length of 'new' when both are lists.")
                # Insert rows in ascending order, adjusting for prior insertions
                order = np.argsort(idx_list)
                for k in order:
                    at = idx_list[k]
                    for key in self.dtypes.keys():
                        col = getattr(self, key)
                        row_block = blocks[key][k:k+1]  # (1,*tail)
                        setattr(self, key, _concat(col, row_block, at))
                    # After inserting one row, subsequent positions >= at must shift by +1
                    idx_list = [i+1 if i >= at else i for i in idx_list]
        else:
            # Single-row create
            for key, dtype in self.dtypes.items():
                col = getattr(self, key)
                val = self.to_default(new.get(key), dtype)
                row_block = _row_block_like(col, val, dtype)
                if index is not None and not isinstance(index, int):
                    raise ValueError("Index must be an int (or None) when creating a single row.")
                setattr(self, key, _concat(col, row_block, index if isinstance(index, int) else None))

        self._recompute_size()

    def read(self, index: int) -> dict[str, Any]:
        """
        Reads all contained data at given index.

        Args:
            index (int): Index at which to read all data

        Returns:
            dict[str, Any]: Dictionary with attribute names as keys and vector elements at given index as values
        """
        return {key: getattr(self, key)[index] for key in self.dtypes}

    def update(self, new: dict[str, Any], index: int):
        """
        Updates data at given index. If any to-be-modified vector is read-only (after finalize), a new copy is created
        first. The original vector is not modified.

        Args:
            new (dict[str, Any]): Dictionary containing attribute names as keys, and their new values
            index (int): Index of row to modify
        """
        for key, value in new.items():
            if key in self.dtypes:
                vector: npt.NDArray = getattr(self, key)
                if not vector.flags.writeable:
                    # Vector is read-only, must copy first.
                    vector = vector.copy()
                    setattr(self, key, vector)
                    vector.flags.writeable = True
                vector[index] = value

    def delete(self, index: int | list[int]):
        """
        Removes data at given index.

        Args:
            index (int | list[int]): Index of row to remove
        """
        for key in self.dtypes:
            vector: npt.NDArray = getattr(self, key)
            setattr(self, key, np.delete(vector, index, axis=0))  # delete always creates a copy

        self._recompute_size()

    def finalize(self):
        """
        Sets all arrays to read-only and returns a shallow copy of self.

        Returns:
            VectorData: Shallow copy of self
        """
        for key in self.dtypes:
            attr: Optional[npt.NDArray]
            attr = getattr(self, key, None)
            if attr is not None:
                attr.flags.writeable = False
        return copy(self)

    def _recompute_size(self) -> None:
        # Find the first present ndarray among declared fields
        for key in self.dtypes:
            arr = getattr(self, key, None)
            if isinstance(arr, np.ndarray):
                self.size = len(arr)
                return
        self.size = 0


class ReferenceTrees(VectorData):
    identifier: npt.NDArray[np.str_]
    tree_number: npt.NDArray[np.int32]
    species: npt.NDArray[np.int32]
    breast_height_diameter: npt.NDArray[np.float64]
    height: npt.NDArray[np.float64]
    measured_height: npt.NDArray[np.float64]
    breast_height_age: npt.NDArray[np.float64]
    biological_age: npt.NDArray[np.float64]
    stems_per_ha: npt.NDArray[np.float64]
    origin: npt.NDArray[np.int32]
    management_category: npt.NDArray[np.int32]
    saw_log_volume_reduction_factor: npt.NDArray[np.float64]
    pruning_year: npt.NDArray[np.int16]
    age_when_10cm_diameter_at_breast_height: npt.NDArray[np.int16]
    stand_origin_relative_position: npt.NDArray[np.float64]
    lowest_living_branch_height: npt.NDArray[np.float64]
    tree_category: npt.NDArray[np.str_]
    storey: npt.NDArray[np.int32]
    sapling: npt.NDArray[np.bool_]
    tree_type: npt.NDArray[np.str_]
    tuhon_ilmiasu: npt.NDArray[np.str_]
    latvuskerros: npt.NDArray[np.float64]

    def __init__(self):
        super().__init__(DTYPES_TREE)

    def as_rst_row(self, i: int) -> list:
        return [
            self.stems_per_ha[i],
            self.species[i],
            self.breast_height_diameter[i],
            self.height[i],
            self.breast_height_age[i],
            self.biological_age[i],
            self.saw_log_volume_reduction_factor[i],
            self.pruning_year[i],
            self.age_when_10cm_diameter_at_breast_height[i],
            self.origin[i],
            self.tree_number[i],
            self.stand_origin_relative_position[i, 0],
            self.stand_origin_relative_position[i, 1],
            self.stand_origin_relative_position[i, 2],
            self.lowest_living_branch_height[i],
            self.management_category[i],
            None,
        ]

    def as_internal_csv_row(self, i) -> list[str]:
        return [
            "tree",
            str(self.identifier[i]),
            str(self.species[i]),
            str(self.origin[i]),
            str(self.stems_per_ha[i]),
            str(self.breast_height_diameter[i]),
            str(self.height[i]),
            str(self.measured_height[i]),
            str(self.breast_height_age[i]),
            str(self.biological_age[i]),
            str(self.saw_log_volume_reduction_factor[i]),
            str(self.pruning_year[i]),
            str(self.age_when_10cm_diameter_at_breast_height[i]),
            str(self.tree_number[i]),
            str(self.stand_origin_relative_position[i, 0]),
            str(self.stand_origin_relative_position[i, 1]),
            str(self.stand_origin_relative_position[i, 2]),
            str(self.lowest_living_branch_height[i]),
            str(self.management_category[i]),
            str(self.tree_category[i]),
            str(self.sapling[i]),
            str(self.storey[i]),
            str(self.tree_type[i]),
            str(self.tuhon_ilmiasu[i])
        ]


class TreeStrata(VectorData):
    identifier: npt.NDArray[np.str_]
    species: npt.NDArray[np.int32]
    mean_diameter: npt.NDArray[np.float64]
    mean_height: npt.NDArray[np.float64]
    breast_height_age: npt.NDArray[np.float64]
    biological_age: npt.NDArray[np.float64]
    stems_per_ha: npt.NDArray[np.float64]
    basal_area: npt.NDArray[np.float64]
    origin: npt.NDArray[np.int32]
    management_category: npt.NDArray[np.int32]
    saw_log_volume_reduction_factor: npt.NDArray[np.float64]
    cutting_year: npt.NDArray[np.int32]
    age_when_10cm_diameter_at_breast_height: npt.NDArray[np.int16]
    tree_number: npt.NDArray[np.int32]
    stand_origin_relative_position: npt.NDArray[np.float64]
    lowest_living_branch_height: npt.NDArray[np.float64]
    storey: npt.NDArray[np.int32]
    sapling_stems_per_ha: npt.NDArray[np.float64]
    sapling_stratum: npt.NDArray[np.bool_]
    number_of_generated_trees: npt.NDArray[np.int32]

    def __init__(self):
        super().__init__(DTYPES_STRATA)

    def as_internal_csv_row(self, i) -> list[str]:
        return [
            "stratum",
            str(self.identifier[i]),
            str(self.species[i]),
            str(self.origin[i]),
            str(self.stems_per_ha[i]),
            str(self.mean_diameter[i]),
            str(self.mean_height[i]),
            str(self.breast_height_age[i]),
            str(self.biological_age[i]),
            str(self.basal_area[i]),
            str(self.saw_log_volume_reduction_factor[i]),
            str(self.cutting_year[i]),
            str(self.age_when_10cm_diameter_at_breast_height[i]),
            str(self.tree_number[i]),
            str(self.stand_origin_relative_position[i, 0]),
            str(self.stand_origin_relative_position[i, 1]),
            str(self.stand_origin_relative_position[i, 2]),
            str(self.lowest_living_branch_height[i]),
            str(self.management_category[i]),
            str(self.sapling_stems_per_ha[i]),
            str(self.sapling_stratum[i]),
            str(self.storey[i])
        ]
