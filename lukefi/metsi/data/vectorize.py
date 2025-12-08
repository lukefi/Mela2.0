from typing import Any
from lukefi.metsi.data.vector_model import ReferenceTrees, TreeStrata
from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.domain.forestry_types import StandList


CONTAINERS = {
    "reference_trees": ReferenceTrees,
    "tree_strata": TreeStrata
}


def vectorize(stands: StandList, **operation_params) -> StandList:
    """
    Modifies a list of ForestStand objects' reference_trees and tree_strata into a struct-of-arrays style.
    The lists of ReferenceTree and TreeStratum are converted into ReferenceTrees and Strata with numpy arrays for
    each attribute.

    Note that this should be the default representation in the future and the conversion from array-of-structs
    should no longer be necessary.

    Args:
        stands (StandList): List of ForestStand objects in standard AoS format

    Returns:
        StandList: A reference to the same list is returned after the objects are modified in-place
    """

    target = operation_params.get('target', None)
    if target is None:
        target = ['reference_trees', 'tree_strata']
    else:
        target = [target]

    for stand in stands:
        for t in target:

            # Get AoS list (may be empty) and existing SoA container (may already be populated)
            pre_list = getattr(stand, f"{t}_pre_vec", None)
            existing_vec = getattr(stand, t, None)

            if (not pre_list) and hasattr(existing_vec, "size") and existing_vec and existing_vec.size > 0:
                continue

            attr_dict: dict[str, Any] = {}
            for data in pre_list or []:
                # Drop back-reference to stand for AoS objects before vectorizing
                if hasattr(data, "stand"):
                    delattr(data, "stand")
                for k, v in data.__dict__.items():
                    attr_dict.setdefault(k, []).append(v)

            container_obj = CONTAINERS.get(t)
            if not container_obj:
                raise MetsiException(f"Unknown target type '{t}'")
            setattr(stand, t, container_obj().vectorize(attr_dict))

            # Only delete *_pre_vec if it actually existed
            if hasattr(stand, f"{t}_pre_vec"):
                delattr(stand, f"{t}_pre_vec")

    return stands


__all__ = ["vectorize"]
