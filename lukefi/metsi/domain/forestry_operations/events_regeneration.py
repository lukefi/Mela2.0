from __future__ import annotations
from typing import Any, Optional
from lukefi.metsi.data.model import ForestStand
from lukefi.metsi.sim.collected_data import OpTuple
from lukefi.metsi.domain.natural_processes.ftrt_regeneration import ftrt_regeneration
from lukefi.metsi.domain.forestry_operations.ftrt_soil_surface_preparation import ftrt_soil_surface_preparation
from lukefi.metsi.domain.forestry_operations.ftrt_cutting import ftrt_cutting
from lukefi.metsi.domain.forestry_operations.ftrt_mark_trees import ftrt_mark_trees

def event_regeneration_chain(
    op: OpTuple[ForestStand],
    *,
    par_trt_rt: dict[str, Any],   # retention tree marking (R-shaped)
    par_trt_cc: dict[str, Any],   # clearcut selection (R-shaped)
    par_trt_ss: dict[str, Any],   # soil surface preparation
    par_trt_pl: dict[str, Any],   # regeneration parameters
    sim_time: Optional[int] = None,
) -> OpTuple[ForestStand]:
    stand, cdata = op

    # 1) retention trees (mark)
    (stand, cdata) = ftrt_mark_trees(
        (stand, cdata),
        tree_selection=par_trt_rt.get("tree_selection"),
        attributes=par_trt_rt.get("attributes", {}),
        labels=par_trt_rt.get("labels", ["retention"]),
        sim_time=sim_time,
    )

    # 2) cutting (clearcut in the example; uses the given selection)
    (stand, cdata) = ftrt_cutting(
        (stand, cdata),
        tree_selection=par_trt_cc.get("tree_selection"),
        labels=par_trt_cc.get("labels", ["clearcutting"]),
        sim_time=sim_time,
    )

    # 3) soil surface preparation
    (stand, cdata) = ftrt_soil_surface_preparation(
        (stand, cdata),
        method=par_trt_ss.get("method"),
        intensity=par_trt_ss.get("intensity"),
        labels=par_trt_ss.get("labels", ["soil_surface_preparation"]),
        sim_time=sim_time,
    )

    # 4) regeneration (planting, by params)
    (stand, cdata) = ftrt_regeneration(
        (stand, cdata),
        origin=par_trt_pl["origin"],
        species=par_trt_pl["species"],
        f=float(par_trt_pl["f"]),
        Hgm=float(par_trt_pl.get("Hgm", 0.7)),
        Dgm=float(par_trt_pl.get("Dgm", 0.0)),
        age_biol=float(par_trt_pl.get("age_biol", 3.0)),
        labels=par_trt_pl.get("labels", ["planting"]),
        sim_time=sim_time,
    )

    return (stand, cdata)
