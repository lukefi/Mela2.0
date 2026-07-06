from abc import ABC, abstractmethod
from copy import copy
import sqlite3
from typing import TYPE_CHECKING, Any, Optional, Type, overload
import numpy as np
import numpy.typing as npt

from lukefi.metsi.sim.exceptions import MetsiException


if TYPE_CHECKING:
    from lukefi.metsi.sim.treatment import PredeterminedTreatment

type DTypeDeclaration = tuple[npt.DTypeLike, Any]


class ComputationalUnit(ABC):
    identifier: str
    time: int = 0
    start_time: int = 0
    predetermined_treatments: list[tuple[int, "PredeterminedTreatment"]] | None

    @abstractmethod
    def output_initial_state_to_db(self, db: sqlite3.Connection):
        pass

    @abstractmethod
    def output_to_db(self, db: sqlite3.Connection, node: str):
        pass

    @abstractmethod
    def update_aggregates(self):
        pass

    @property
    def relative_time(self):
        return self.time - self.start_time

    @classmethod
    @abstractmethod
    def reconstruct_initial_state[T: "ComputationalUnit"](cls: Type[T], identifier: str, db: sqlite3.Connection) -> T:
        pass

    @classmethod
    @abstractmethod
    def create_database_tables(cls, db: sqlite3.Connection, sqlite_decl: dict[str, str] | None = None):
        pass


class Finalizable(ABC):
    @abstractmethod
    def finalize[T: "Finalizable"](self: T) -> T:
        pass


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
