""" Operations related for manipulating the exporting related formats.
NOTE: Only for pipeline component 'export_prepro' """
import numpy as np
from lukefi.metsi.domain.forestry_types import StandList
from lukefi.metsi.data.conversion.internal2mela import mela_stand
from lukefi.metsi.app.utils import ConfigurationException


def _recreate_stand_indices(stands: StandList) -> StandList:
    for idx, stand in enumerate(stands):
        stand.set_identifiers(idx + 1)
    return stands


def prepare_rst_output(stands: StandList, **operation_params) -> StandList:
    """
    Recreate forest stands for output in SoA form:
        1) filter out non-living reference trees (on stand.reference_trees)
        2) recreate indices for reference trees (tree_number 1..N)
        3) filter out non-forestland stands and empty auxiliary stands
        4) recreate indices for stands
    """
    _ = operation_params

    for stand in stands:
        trees = stand.reference_trees

        # Nothing to do if there are no trees yet
        if trees.size == 0:
            continue

        # SoA version of ReferenceTree.is_living
        living_mask = np.isin(
            trees.tree_category,
            np.array(["", "0", "1", "3", "7"], dtype=trees.tree_category.dtype)
        )

        # Filter to living trees only (VectorBase.__getitem__ handles boolean masks)
        trees = trees[living_mask]

        # Recreate tree_number: 1..N
        if trees.size > 0:
            trees.tree_number = np.arange(1, trees.size + 1, dtype=trees.tree_number.dtype)

        stand.reference_trees = trees

    # Keep only:
    #  - forest land, and
    #  - either non-auxiliary or auxiliary with trees or strata
    stands = [
        s for s in stands
        if (s.is_forest_land() and (not s.is_auxiliary() or s.has_trees() or s.has_strata()))
    ]

    # Recreate stand indices as before
    stands = _recreate_stand_indices(stands)
    return stands


def classify_values_to(stands: StandList, **operation_params) -> StandList:
    """ Give format as parameter """
    format_ = operation_params.get('format', None)
    if format_ not in ('rst', 'rsts'):
        raise ConfigurationException(f"unsupported format: {format_}")
    return [mela_stand(stand) for stand in stands]
