from copy import copy
from dataclasses import dataclass
from typing import Any, Optional, overload
import numpy as np
import numpy.typing as npt

from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.enums.internal import (
    CrownClass,
    DamageType,
    Origin,
    Storey,
    StratumRank,
    TreeCategory,
    TreeManagementCategory,
    TreeSpecies,
    TreeType)
from lukefi.metsi.data.formats.util import convert_str_to_type as conv
type DTypeDeclaration = tuple[npt.DTypeLike, Any]

DTYPES_TREE: dict[str, DTypeDeclaration] = {
    "identifier": (np.dtype("U30"), ""),
    "tree_number": (np.int32, -1),
    "species": (np.int32, TreeSpecies.UNSET),
    "breast_height_diameter": (np.float64, 0.0),
    "height": (np.float64, np.nan),
    "measured_height": (np.float64, np.nan),
    "breast_height_age": (np.float64, np.nan),
    "biological_age": (np.float64, np.nan),
    "stems_per_ha": (np.float64, 0.0),
    "origin": (np.int32, Origin.UNSET),
    "management_category": (np.int32, TreeManagementCategory.UNSET),
    "tree_category": (np.dtype("U1"), TreeCategory.UNSET),
    "storey": (np.int32, Storey.UNSET),
    "sapling": (np.bool_, False),
    "tree_type": (np.dtype("U1"), TreeType.UNSET),
    "damage_type": (np.dtype("U2"), DamageType.UNSET),
    "crown_class": (np.dtype("U1"), CrownClass.UNSET),
    "basal_area": (np.float64, 0.0),
    "volume": (np.float64, 0.0),
    "stratum": (np.int32, -1)
}

DTYPES_STRATA: dict[str, DTypeDeclaration] = {
    "identifier": (np.dtype("U30"), ""),
    "species": (np.int32, TreeSpecies.UNSET),
    "mean_diameter": (np.float64, -1),
    "mean_height": (np.float64, 0.0),
    "breast_height_age": (np.float64, np.nan),
    "biological_age": (np.float64, np.nan),
    "stems_per_ha": (np.float64, 0.0),
    "basal_area": (np.float64, np.nan),
    "origin": (np.int32, Origin.UNSET),
    "stratum_number": (np.int32, -1),
    "storey": (np.int32, Storey.UNSET),
    "sapling_stems_per_ha": (np.float64, 0.0),
    "number_of_generated_trees": (np.int32, -1),
    "stratum_rank": (np.int16, StratumRank.UNSET)
}

TREE_INTERNAL_CSV_COLUMNS = (
    "identifier",
    "species",
    "origin",
    "stems_per_ha",
    "breast_height_diameter",
    "height",
    "measured_height",
    "breast_height_age",
    "biological_age",
    "tree_number",
    "management_category",
    "tree_category",
    "sapling",
    "storey",
    "tree_type",
    "damage_type",
)

STRATUM_INTERNAL_CSV_COLUMNS = (
    "identifier",
    "species",
    "origin",
    "stems_per_ha",
    "mean_diameter",
    "mean_height",
    "breast_height_age",
    "biological_age",
    "basal_area",
    "stratum_number",
    "sapling_stems_per_ha",
    "storey",
)

_ENUM_FIELDS = {
    "species": TreeSpecies,
    "origin": Origin,
    "storey": Storey,
}

_BOOL_FIELDS = {"sapling"}


def _parse_csv_cell(field_name: str, raw: str) -> Any:
    if raw == "None":
        return None

    if field_name in _BOOL_FIELDS:
        return raw == "True"

    if field_name in _ENUM_FIELDS:
        return _ENUM_FIELDS[field_name](int(raw)).value

    if field_name in DTYPES_TREE:
        dtype = DTYPES_TREE[field_name][0]
    elif field_name in DTYPES_STRATA:
        dtype = DTYPES_STRATA[field_name][0]
    else:
        return raw

    if dtype in (np.int32, np.int16):
        return conv(raw, int)
    if dtype == np.float64:
        return conv(raw, float)
    if dtype == np.bool_:
        return raw == "True"
    return conv(raw, str)


def attrs_from_internal_tree_csv_row(row: list[str]) -> dict[str, Any]:
    assert row[0] == "tree"
    values = row[1:1 + len(TREE_INTERNAL_CSV_COLUMNS)]
    return {
        field_name: _parse_csv_cell(field_name, raw)
        for field_name, raw in zip(TREE_INTERNAL_CSV_COLUMNS, values)
    }


def attrs_from_internal_stratum_csv_row(row: list[str]) -> dict[str, Any]:

    assert row[0] == "stratum"
    values = row[1:1 + len(STRATUM_INTERNAL_CSV_COLUMNS)]
    return {
        field_name: _parse_csv_cell(field_name, raw)
        for field_name, raw in zip(STRATUM_INTERNAL_CSV_COLUMNS, values)
    }


class VectorData():
    """
    Base class for generic SoA data.
    """
    dtypes: dict[str, DTypeDeclaration]
    size: int

    def __init__(self, dtypes: dict[str, DTypeDeclaration], size: int = 0):
        self.dtypes = dtypes
        self.vectorize({"identifier": [""] * size})

    def __len__(self):
        return self.size

    @overload
    def __getitem__(self, name: str) -> npt.NDArray:
        pass

    @overload
    def __getitem__[T: "VectorData"](self: T, name: slice | npt.NDArray) -> T:
        pass

    def __getitem__[T: "VectorData"](self: T, name: str | slice | npt.NDArray) -> npt.NDArray | T:
        if isinstance(name, str):
            return getattr(self, name)

        retval = copy(self)
        for attribute in self.dtypes.keys():
            setattr(retval, attribute, getattr(self, attribute)[name])
        retval._recompute_size()
        return retval

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
                    data_type[0]))
            if not self.is_contiguous(attribute_name):
                raise MetsiException("Vectorized data is not contiguous")
        return self

    def is_contiguous(self, name: str):
        arr: npt.NDArray = getattr(self, name)
        return bool(arr.flags['CONTIGUOUS']) and bool(arr.flags['C_CONTIGUOUS'])

    def set_size(self, attr_dict: dict[str, list[Any]]):
        size = len(attr_dict.get('identifier', []))
        setattr(self, 'size', size)

    def defaultify(self, values: list, dtype: DTypeDeclaration) -> list:
        return [self.to_default(v, dtype) for v in values]

    def to_default(self, value: Optional[Any], field_type: DTypeDeclaration) -> Any:
        """ Replace None with appropriate defaults based on field type. """
        return value if value is not None else field_type[1]

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
        if isinstance(new, list):
            for key, dtype in self.dtypes.items():
                values = [self.to_default(new_item.get(key), dtype) for new_item in new]
                vector: npt.NDArray = getattr(self, key)
                if index is not None:
                    setattr(self, key, np.insert(vector, index, values, axis=0))  # insert always creates a copy
                else:
                    setattr(self, key, np.append(vector, values, axis=0))  # append always creates a copy
        else:
            for key, dtype in self.dtypes.items():
                value = self.to_default(new.get(key), dtype)
                vector = getattr(self, key)
                if index is not None:
                    setattr(self, key, np.insert(vector, index, value, axis=0))  # insert always creates a copy
                else:
                    setattr(self, key, np.append(vector, [value], axis=0))  # append always creates a copy

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

    def update_many(
        self,
        new: dict[str, Any | npt.NDArray | list[Any]],
        index: int | list[int] | npt.NDArray[np.int_] | npt.NDArray[np.bool_],
    ):
        """
        Updates multiple rows at once using vectorized indexing. If any to-be-modified vector is read-only
        (after finalize), a new copy is created first. The original vector is not modified.

        Supports scalar updates (broadcasted to all selected indices) as well as per-index updates via
        arrays or lists.

        Args:
            new (dict[str, Any | ndarray | list]): Dictionary containing attribute names as keys, and
                their new values. Values can be scalars, lists, or numpy arrays.
            index (int | list[int] | ndarray[int] | ndarray[bool]): Index or indices of rows to modify.
                Can be a single index, a list/array of indices, or a boolean mask.
        """
        for key, value in new.items():
            if key not in self.dtypes:
                continue

            vector: npt.NDArray = getattr(self, key)
            if not vector.flags.writeable:
                vector = vector.copy()
                setattr(self, key, vector)
                vector.flags.writeable = True

            vector[index] = value

    def delete(self, index: int | list[int] | npt.NDArray[np.int_]):
        """
        Removes data at given index/indices.

        Args:
            index: Row index/indices to remove. May be an int, list[int], or numpy integer array
                (e.g. output of np.where(mask)[0]).
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


@dataclass
class ReferenceTree:
    identifier: str = ""
    tree_number: int = -1
    species: TreeSpecies = TreeSpecies.UNSET
    breast_height_diameter: float = 0.0
    height: Optional[float] = None
    measured_height: Optional[float] = None
    breast_height_age: Optional[float] = None
    biological_age: Optional[float] = None
    stems_per_ha: float = 0.0
    origin: Origin = Origin.UNSET
    management_category: TreeManagementCategory = TreeManagementCategory.UNSET
    tree_category: TreeCategory = TreeCategory.UNSET
    storey: Storey = Storey.UNSET
    sapling: bool = False
    tree_type: TreeType = TreeType.UNSET
    damage_type: DamageType = DamageType.UNSET
    crown_class: CrownClass = CrownClass.UNSET
    basal_area: float = 0.0
    volume: float = 0.0


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
    tree_category: npt.NDArray[np.str_]
    storey: npt.NDArray[np.int32]
    sapling: npt.NDArray[np.bool_]
    tree_type: npt.NDArray[np.str_]
    damage_type: npt.NDArray[np.str_]
    crown_class: npt.NDArray[np.str_]
    basal_area: npt.NDArray[np.float64]
    volume: npt.NDArray[np.float64]
    stratum: npt.NDArray[np.int32]

    def __init__(self, size: int = 0):
        super().__init__(DTYPES_TREE, size)

    def __add__(self, other: "ReferenceTrees") -> "ReferenceTrees":
        retval = ReferenceTrees()
        for attribute_name in self.dtypes.keys():
            setattr(retval, attribute_name, np.concat((getattr(self, attribute_name),
                                                       getattr(other, attribute_name)), axis=0))
        retval._recompute_size()
        return retval

    def __repr__(self) -> str:
        return f"ReferenceTrees(size={self.size})"

    def get_tree(self, i: int) -> ReferenceTree:
        return ReferenceTree(
            self.identifier[i],
            self.tree_number[i],
            TreeSpecies(self.species[i]),
            self.breast_height_diameter[i],
            self.height[i] if not np.isnan(self.height[i]) else None,
            self.measured_height[i] if not np.isnan(self.measured_height[i]) else None,
            self.breast_height_age[i] if not np.isnan(self.breast_height_age[i]) else None,
            self.biological_age[i] if not np.isnan(self.biological_age[i]) else None,
            self.stems_per_ha[i],
            Origin(self.origin[i]),
            TreeManagementCategory(self.management_category[i]),
            TreeCategory(self.tree_category[i]),
            Storey(self.storey[i]),
            self.sapling[i],
            TreeType(self.tree_type[i]),
            DamageType(self.damage_type[i]),
            CrownClass(self.crown_class[i]),
            self.basal_area[i],
            self.volume[i]
        )

    def as_rst_row(self, i: int) -> list:
        return [
            self.stems_per_ha[i],
            self.species[i],
            self.breast_height_diameter[i],
            self.height[i],
            self.breast_height_age[i],
            self.biological_age[i],
            None,
            None,
            None,
            self.origin[i],
            self.tree_number[i],
            None,
            None,
            None,
            None,
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
            str(self.tree_number[i]),
            str(self.management_category[i]),
            str(self.tree_category[i]),
            str(self.sapling[i]),
            str(self.storey[i]),
            str(self.tree_type[i]),
            str(self.damage_type[i])
        ]


@dataclass
class TreeStratum:
    identifier: str = ""
    species: TreeSpecies = TreeSpecies.UNSET
    mean_diameter: float = -1
    mean_height: float = 0.0
    breast_height_age: Optional[float] = None
    biological_age: Optional[float] = None
    stems_per_ha: float = 0.0
    basal_area: Optional[float] = None
    origin: Origin = Origin.UNSET
    stratum_number: int = 0
    storey: Storey = Storey.UNSET
    sapling_stems_per_ha: float = 0.0
    number_of_generated_trees: int = 0
    stratum_rank: StratumRank = StratumRank.UNSET

    def get_breast_height_age(self, subtrahend: float = 12.0) -> float:
        if self.breast_height_age is not None and self.breast_height_age > 0.0:
            return self.breast_height_age
        if self.biological_age is not None and self.biological_age > 0.0:
            new_breast_height_age = self.biological_age - subtrahend
            return 0.0 if new_breast_height_age <= 0.0 else new_breast_height_age
        return 0.0


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
    stratum_number: npt.NDArray[np.int32]
    storey: npt.NDArray[np.int32]
    sapling_stems_per_ha: npt.NDArray[np.float64]
    number_of_generated_trees: npt.NDArray[np.int32]
    stratum_rank: npt.NDArray[np.int16]

    def __init__(self, size: int = 0):
        super().__init__(DTYPES_STRATA, size)

    def __add__(self, other: "TreeStrata") -> "TreeStrata":
        retval = TreeStrata()
        for attribute_name in self.dtypes.keys():
            setattr(retval, attribute_name, np.concat((getattr(self, attribute_name),
                                                       getattr(other, attribute_name)), axis=0))
        retval._recompute_size()
        return retval

    def __repr__(self) -> str:
        return f"TreeStrata(size={self.size})"

    def get_stratum(self, i: int) -> TreeStratum:
        return TreeStratum(
            self.identifier[i],
            TreeSpecies(self.species[i]),
            self.mean_diameter[i],
            self.mean_height[i],
            self.breast_height_age[i] if not np.isnan(self.breast_height_age[i]) else None,
            self.biological_age[i] if not np.isnan(self.biological_age[i]) else None,
            self.stems_per_ha[i],
            self.basal_area[i] if not np.isnan(self.basal_area[i]) else None,
            Origin(self.origin[i]),
            self.stratum_number[i],
            Storey(self.storey[i]),
            self.sapling_stems_per_ha[i],
            self.number_of_generated_trees[i],
            StratumRank(self.stratum_rank[i])
        )

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
            str(self.stratum_number[i]),
            str(self.sapling_stems_per_ha[i]),
            str(self.storey[i])
        ]
