from lukefi.metsi.sim.operations import do_nothing
from lukefi.metsi.sim.treatment import PredeterminedTreatment
# from lukefi.metsi.domain.forestry_treatments.earlycare import earlycare_fn
# from lukefi.metsi.domain.forestry_treatments.pct import pct_fn

# TODO: earlycare only implemented in Motti
mechanicalClearing = PredeterminedTreatment(
    name="fc_mechanical_clearing",
    # treatment_fn=earlycare_fn,
    treatment_fn=do_nothing,
    static_parameters={
        "imode": 1
    }
)

# TODO: pct only implemented in Motti
earlyTending = PredeterminedTreatment(
    name="fc_mechanical_clearing",
    # treatment_fn=pct_fn,
    treatment_fn=do_nothing
)
