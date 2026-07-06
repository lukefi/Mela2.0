from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import numpy.typing as npt

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
from lukefi.metsi.sim.model import DTypeDeclaration, VectorData

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


@dataclass
class ReferenceTree:
    identifier: str = ""
    """
    Free-form tree identifier. Each tree must have a unique identifier,
    even between different forest stands.
    """
    tree_number: int = -1
    """
    Running number for generated trees. The combined `tree_number`
    and `stratum` must be unique within a single forest stand.
    """
    species: TreeSpecies = TreeSpecies.UNSET
    """
    Species of the tree.
    """
    breast_height_diameter: float = 0.0
    """
    Diameter of the tree at 1.3 m height [cm].
    """
    height: Optional[float] = None
    """
    Height of the tree [m].
    """
    measured_height: Optional[float] = None
    """
    Measured height of the tree [m].
    """
    breast_height_age: Optional[float] = None
    """
    Age of the tree at breast height [a]. (Years since the tree reached breast height)
    """
    biological_age: Optional[float] = None
    """
    Biological age of the tree [a]. (Years since the tree was born)
    """
    stems_per_ha: float = 0.0
    """
    Number of stems per hectare that the tree represents [1/ha].
    """
    origin: Origin = Origin.UNSET
    """
    Origin of the tree.
    """
    management_category: TreeManagementCategory = TreeManagementCategory.UNSET
    """
    Management category of the tree.
    """
    tree_category: TreeCategory = TreeCategory.UNSET
    """
    NFI tree category for living/dead/otherwise unusable tree.
    """
    storey: Storey = Storey.UNSET
    """
    Storey of the tree.
    """
    sapling: bool = False
    """
    Whether the tree is considered a sapling or adult.
    """
    tree_type: TreeType = TreeType.UNSET
    """
    Type of the tree (old, new, remeasured, etc.).
    """
    damage_type: DamageType = DamageType.UNSET
    """
    Type of damage affecting the tree.
    """
    crown_class: CrownClass = CrownClass.UNSET
    """
    Tree crown class.
    """
    basal_area: float = 0.0
    """
    Basal area of singe tree of this type [m^2].
    """
    volume: float = 0.0
    """
    Volume of single tree of this type [m^3].
    """


class ReferenceTrees(VectorData):
    identifier: npt.NDArray[np.str_]
    """
    Free-form tree identifier. Each tree must have a unique identifier,
    even between different forest stands.
    """
    tree_number: npt.NDArray[np.int32]
    """
    Running number for generated trees. The combined `tree_number`
    and `stratum` must be unique within a single forest stand.
    """
    species: npt.NDArray[np.int32]
    """
    Species of the tree.
    """
    breast_height_diameter: npt.NDArray[np.float64]
    """
    Diameter of the tree at 1.3 m height [cm].
    """
    height: npt.NDArray[np.float64]
    """
    Height of the tree [m].
    """
    measured_height: npt.NDArray[np.float64]
    """
    Measured height of the tree [m].
    """
    breast_height_age: npt.NDArray[np.float64]
    """
    Age of the tree at breast height [a]. (Years since the tree reached breast height)
    """
    biological_age: npt.NDArray[np.float64]
    """
    Biological age of the tree [a]. (Years since the tree was born)
    """
    stems_per_ha: npt.NDArray[np.float64]
    """
    Number of stems per hectare that the tree represents [1/ha].
    """
    origin: npt.NDArray[np.int32]
    """
    Origin of the tree.
    """
    management_category: npt.NDArray[np.int32]
    """
    Management category of the tree.
    """
    tree_category: npt.NDArray[np.str_]
    """
    NFI tree category for living/dead/otherwise unusable tree.
    """
    storey: npt.NDArray[np.int32]
    """
    Storey of the tree.
    """
    sapling: npt.NDArray[np.bool_]
    """
    Whether the tree is considered a sapling or adult.
    """
    tree_type: npt.NDArray[np.str_]
    """
    Type of the tree (old, new, remeasured, etc.).
    """
    damage_type: npt.NDArray[np.str_]
    """
    Type of damage affecting the tree.
    """
    crown_class: npt.NDArray[np.str_]
    """
    Tree crown class.
    """
    basal_area: npt.NDArray[np.float64]
    """
    Basal area of singe tree of this type [m^2].
    """
    volume: npt.NDArray[np.float64]
    """
    Volume of single tree of this type [m^3].
    """
    stratum: npt.NDArray[np.int32]
    """
    `stratum_number` of the stratum this tree is related to.
    """

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
    """
    Free-form stratum identifier. Each stratum must have a unique identifier,
    even between different forest stands.
    """
    species: TreeSpecies = TreeSpecies.UNSET
    """
    Main tree species of the stratum.
    """
    mean_diameter: float = -1
    """
    Mean diameter of trees in the stratum [cm].
    """
    mean_height: float = 0.0
    """
    Mean height of trees in the stratum [m].
    """
    breast_height_age: Optional[float] = None
    """
    Age of the tree at breast height [a]. (Years since the tree reached breast height)
    """
    biological_age: Optional[float] = None
    """
    Biological age of the tree [a]. (Years since the tree was born)
    """
    stems_per_ha: float = 0.0
    """
    Number of stems in area of one hectare [1/ha].
    """
    basal_area: Optional[float] = None
    """
    Basal area of the stratum [m^2].
    """
    origin: Origin = Origin.UNSET
    """
    Origin of the stratum.
    """
    stratum_number: int = 0
    """
    Running stratum number within the forest stand.
    """
    storey: Storey = Storey.UNSET
    """
    Storey of the stratum.
    """
    sapling_stems_per_ha: float = 0.0
    """
    Number of sapling stems in area of one hectare [1/ha].
    """
    number_of_generated_trees: int = 0
    """
    Number of reference trees generated from this stratum.
    """
    stratum_rank: StratumRank = StratumRank.UNSET
    """
    Rank of the stratum.
    """

    def get_breast_height_age(self, subtrahend: float = 12.0) -> float:
        if self.breast_height_age is not None and self.breast_height_age > 0.0:
            return self.breast_height_age
        if self.biological_age is not None and self.biological_age > 0.0:
            new_breast_height_age = self.biological_age - subtrahend
            return 0.0 if new_breast_height_age <= 0.0 else new_breast_height_age
        return 0.0


class TreeStrata(VectorData):
    identifier: npt.NDArray[np.str_]
    """
    Free-form stratum identifier. Each stratum must have a unique identifier,
    even between different forest stands.
    """
    species: npt.NDArray[np.int32]
    """
    Main tree species of the stratum.
    """
    mean_diameter: npt.NDArray[np.float64]
    """
    Mean diameter of trees in the stratum [cm].
    """
    mean_height: npt.NDArray[np.float64]
    """
    Mean height of trees in the stratum [m].
    """
    breast_height_age: npt.NDArray[np.float64]
    """
    Age of the tree at breast height [a]. (Years since the tree reached breast height)
    """
    biological_age: npt.NDArray[np.float64]
    """
    Biological age of the tree [a]. (Years since the tree was born)
    """
    stems_per_ha: npt.NDArray[np.float64]
    """
    Number of stems in area of one hectare [1/ha].
    """
    basal_area: npt.NDArray[np.float64]
    """
    Basal area of the stratum [m^2].
    """
    origin: npt.NDArray[np.int32]
    """
    Origin of the stratum.
    """
    stratum_number: npt.NDArray[np.int32]
    """
    Running stratum number within the forest stand.
    """
    storey: npt.NDArray[np.int32]
    """
    Storey of the stratum.
    """
    sapling_stems_per_ha: npt.NDArray[np.float64]
    """
    Number of sapling stems in area of one hectare [1/ha].
    """
    number_of_generated_trees: npt.NDArray[np.int32]
    """
    Number of reference trees generated from this stratum.
    """
    stratum_rank: npt.NDArray[np.int16]
    """
    Rank of the stratum.
    """

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
