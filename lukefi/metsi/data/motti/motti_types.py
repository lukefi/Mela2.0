from lukefi.metsi.app.utils import MetsiException


class MottiStub:
    def __init__(self) -> None:
        raise MetsiException("This is a typing stub class that should never be initialized")


class Motti4Spev9(MottiStub):
    ma: float
    ku: float
    ra: float
    hi: float
    ha: float
    hl: float
    tl: float
    mh: float
    ml: float


class Motti4Spev10(MottiStub):
    total: float
    ma: float
    ku: float
    ra: float
    hi: float
    ha: float
    hl: float
    tl: float
    mh: float
    ml: float


class Motti4StoreyInfo(MottiStub):
    age100: float
    h100: float
    g: float
    f: float
    dg: float
    spe: float


class Motti4Site(MottiStub):
    Y: float
    X: float
    Z: float
    lake: float
    sea: float
    dd: float
    _7: float
    _8: float
    _9: float
    _10: float
    rgn_nat_spe: float
    rgn_seedratio: float
    rgn_vlj_spe: float
    rgn_f: float
    rgn_surv: float
    _16: float
    xt_regen: float
    xt_muok: float
    xt_raiv: float
    xt_fert_prev: float
    mal: float
    mty: float
    verl: float
    verlt: float
    alr: float
    pd: float
    _27: float
    muok: float
    _29: float
    _30: float
    si: Motti4Spev9
    tkg: float
    hd50: Motti4Spev9
    year: float
    step: float
    sid: float
    _53: float
    xt_perk: float
    prt: float
    fthin: float
    xt_thin: float
    xt_fert: float
    fert_peat: float
    xt_thoit: float
    drain: float
    xt_ndrain: float
    xt_rdrain: float
    _64: float
    _65: float
    _66: float
    _67: float
    _68: float
    xt_kar: float
    xt_fthin: float
    agedom: float
    agedom13: float
    ndom: float
    spedom: float
    spedom2: float
    dcond: float
    kehl: float
    nstorey: float
    gstorey: float
    st1: Motti4StoreyInfo
    st2: Motti4StoreyInfo
    st3: Motti4StoreyInfo
    st4: Motti4StoreyInfo
    st12: Motti4StoreyInfo
    hdom100: Motti4Spev10
    hdom_j: Motti4Spev10
    hg: Motti4Spev10
    hf: Motti4Spev10
    ddom100: Motti4Spev10
    ddom_latv: Motti4Spev10
    dg: Motti4Spev10
    df: Motti4Spev10
    h100_perk: float
    crdom: float
    crerror: float
    rimp: float
    vg: float
    v1: float
    v2: float
    v3: float
    v4: float
    v12: float
    f: Motti4Spev10
    f13: Motti4Spev10
    G: Motti4Spev10
    ccf: Motti4Spev10
    _240: list[float]
    ccfi: Motti4Spev10
    V: Motti4Spev10
    f_dead: float
    ba_dead: float
    v_dead: float
    _273: float
    _274: float
    _275: float
    _276: float
    _277: float
    jh: float
    jd: float
    xhdom: Motti4Spev10
    _290: float
    ddomg0: float
    dgdom0: float
    ba0: float
    h100_0: float
    cr100_0: float
    v0: float
    dg0: float
    _298: float
    dgM: float
    _300: float
    _yy2: list[float]


class Motti4Biomass(MottiStub):
    trunk: float
    waste: float
    branch_live: float
    branch_dead: float
    leaf: float
    base: float
    root_dense: float
    root_thin: float


class Motti4Tree(MottiStub):
    id: float
    f: float
    spe: float
    age: float
    age13: float
    d13: float
    h: float
    cr: float
    snt: float
    ccftop: Motti4Spev10
    bal: Motti4Spev10
    vol: float
    vol_t: float
    vol_s: float
    vol_f: float
    waste: float
    destr: float
    crfix: float
    keh: float
    storie: float
    latraj: float
    crerror: float
    h0: float
    cr0: float
    crt: float
    d13_0: float
    sid: float
    fdead: float
    xd: float
    xg: float
    xh: float
    xvol: float
    xvol_dead: float
    ba: float
    _53: float
    thin1: float
    thin2: float
    _56: list[float]
    bm: Motti4Biomass
    _89: list[float]


type Motti4Trees = list[list[Motti4Tree]]


class Motti4SaplingStratum(MottiStub):
    year: float
    age: float
    hdom: float
    f_kkp: float
    f_klv: float
    f_vlj: float
    crfix_kkp: float
    crfix_klv: float
    crfix_vlj: float
    N_kkp: float
    N_klv: float
    N_vlj: float
    h_kkp: float
    h_klv: float
    h_vlj: float
    d_kkp: float
    d_klv: float
    d_vlj: float
    osid_kkp: float
    osid_klv: float
    osid_vlj: float
    age_kkp: float
    age_klv: float
    age_vlj: float
    age13_kkp: float
    age13_klv: float
    age13_vlj: float
    g_kkp: float
    g_klv: float
    g_vlj: float
    v_kkp: float
    v_klv: float
    v_vlj: float
    hg_kkp: float
    hg_klv: float
    hg_vlj: float
    dg_kkp: float
    dg_klv: float
    dg_vlj: float
    _40: float


class Motti4SaplingsSpev(MottiStub):
    ma: Motti4SaplingStratum
    ku: Motti4SaplingStratum
    ra: Motti4SaplingStratum
    hi: Motti4SaplingStratum
    ha: Motti4SaplingStratum
    hl: Motti4SaplingStratum
    tl: Motti4SaplingStratum
    mh: Motti4SaplingStratum
    ml: Motti4SaplingStratum
    _10: Motti4SaplingStratum


type Motti4Saplings = list[list[Motti4SaplingsSpev]]
type Motti4KorArray = list[list[float]]
type Motti4VcrArray = list[list[float]]


class Motti4Fertilization(MottiStub):
    year: float
    V0: float
    N0: float
    alr: float
    _5: float
    type: float
    amount: float
    p: float
    phos: float
    _10: float


type Motti4FerArray = list[list[Motti4Fertilization]]
type IntPtr = list[int]
type FloatPtr = list[float]

class Motti4Ctrl(MottiStub):
    death_forest: int
    death_tree: int
    _3: int
    _4: int
    _5: int
    _6: int
    _7: int
    _8: int
    calibrate: int
    _10: int

class Motti4Stratum(MottiStub):
    spe: float
    age: float
    ba: float
    f: float
    h: float
    hw: float
    d: float
    dg: float
    storey: float
    st: float
    sid: float

type Motti4Strata = list[list[Motti4Stratum]]
