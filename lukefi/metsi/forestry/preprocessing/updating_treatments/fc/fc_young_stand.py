from lukefi.metsi.sim.treatment import PredeterminedTreatment
from lukefi.metsi.domain.forestry_treatments.earlycare import earlycare
from lukefi.metsi.domain.forestry_treatments.pct import pct


mechanicalClearing = PredeterminedTreatment(
    name="fc_mechanical_clearing",
    treatment_fn=earlycare,
    static_parameters={
        "imode": 1
    }
)

earlyTending = PredeterminedTreatment(
    name="fc_mechanical_clearing",
    treatment_fn=pct,
)
