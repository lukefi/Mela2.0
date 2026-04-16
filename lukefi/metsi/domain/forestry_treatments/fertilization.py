from lukefi.metsi.app.utils import MetsiException
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.sim.treatment import Treatment
from lukefi.metsi.domain.natural_processes.grow_motti_dll import (
    sync_ut_to_reference_trees,
    sync_yp_to_reference_trees,
    prune_reference_trees_not_in_motti,
)


def mineral_soils_fertilization_fn(input_: ForestStand, /, **operation_parameters) -> OpTuple[ForestStand]:
    """
    Motti-only mineral-soils fertilization treatment.

    Parameters
    ----------
    ftype : int
        Fertilization type code passed through to Motti.
    amount_n : float
        Nitrogen amount. Alias: amountN
    bool_phosphorus : int
        0/1 flag indicating whether phosphorus is included.
        Aliases: boolPhosporus, phosphorus
    """
    stand = input_

    ms = getattr(stand, "motti_state", None)
    if ms is None or ms.buffers is None:
        raise MetsiException(
            "Motti mineral-soils fertilization requested but stand has no initialized motti_state. "
        )

    if "ftype" not in operation_parameters:
        raise MetsiException("Fertilization parameter 'ftype' is required")

    amount_n = operation_parameters.get("amount_n", operation_parameters.get("amountN"))
    if amount_n is None:
        raise MetsiException("Fertilization parameter 'amount_n' (or amountN) is required")

    bool_phosphorus = operation_parameters.get(
        "bool_phosphorus",
        operation_parameters.get("boolPhosporus", operation_parameters.get("phosphorus", 0)),
    )

    _response = ms.dll.mineral_soils_fertilization_with_state(
        ms.yy,
        ms.yp,
        int(ms.ntrees),
        ms.buffers,
        ftype=int(operation_parameters["ftype"]),
        amount_n=float(amount_n),
        bool_phosphorus=int(bool(bool_phosphorus)),
    )

    # Keep Python-side vectors aligned in case the DLL updated state immediately.
    sync_yp_to_reference_trees(stand)
    sync_ut_to_reference_trees(stand)
    prune_reference_trees_not_in_motti(stand)

    stand.fertilization_year = stand.year

    return stand, []


mineral_soils_fertilization = Treatment(mineral_soils_fertilization_fn, "mineral_soils_fertilization")
