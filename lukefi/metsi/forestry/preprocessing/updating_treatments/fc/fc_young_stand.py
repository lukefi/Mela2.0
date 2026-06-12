from lukefi.metsi.sim.treatment import PredeterminedTreatment
from lukefi.metsi.domain.forestry_treatments.earlycare import earlycare_fn
from lukefi.metsi.domain.forestry_treatments.pct import pct_fn


mechanicalClearing = PredeterminedTreatment(
    name="fc_mechanical_clearing",
    treatment_fn=earlycare_fn,
    static_parameters={
        "imode": 1
    }
)

earlyTending = PredeterminedTreatment(
    name="fc_mechanical_clearing",
    treatment_fn=pct_fn,
)
