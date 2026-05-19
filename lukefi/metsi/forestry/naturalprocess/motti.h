typedef struct
{
    float ma, ku, ra, hi, ha, hl, tl, mh, ml;
} Motti4Spev9;

typedef struct
{
    float total, ma, ku, ra, hi, ha, hl, tl, mh, ml;
} Motti4Spev10;

typedef struct
{
    float trunk, waste, branch_live, branch_dead, leaf, base, root_dense, root_thin;
} Motti4Biomass;

typedef struct
{
    float id;
    float f;
    float spe;
    float age;
    float age13;
    float d13;
    float h;
    float cr;
    float snt;
    Motti4Spev10 ccftop;
    Motti4Spev10 bal;
    float vol;
    float vol_t;
    float vol_s;
    float vol_f;
    float waste;
    float destr;
    float crfix;
    float keh;
    float storie;
    float latraj;
    float crerror;
    float h0;
    float cr0;
    float crt;
    float d13_0;
    float sid;
    float fdead;
    float xd;
    float xg;
    float xh;
    float xvol;
    float xvol_dead;
    float ba;
    float _53;
    float thin1;
    float thin2;
    float _56[25];
    Motti4Biomass bm;
    float _89[2];
} Motti4Tree;

typedef Motti4Tree Motti4Trees[1000];

typedef struct
{
    float spe, age, ba, f, h, hw, d, dg, storey, st, sid;
} Motti4Stratum;

typedef Motti4Stratum Motti4Strata[10];

typedef struct
{
    float year, age, hdom;
    float f_kkp, f_klv, f_vlj;
    float crfix_kkp, crfix_klv, crfix_vlj;
    float N_kkp, N_klv, N_vlj;
    float h_kkp, h_klv, h_vlj;
    float d_kkp, d_klv, d_vlj;
    float osid_kkp, osid_klv, osid_vlj;
    float age_kkp, age_klv, age_vlj;
    float age13_kkp, age13_klv, age13_vlj;
    float g_kkp, g_klv, g_vlj;
    float v_kkp, v_klv, v_vlj;
    float hg_kkp, hg_klv, hg_vlj;
    float dg_kkp, dg_klv, dg_vlj;
    float _40;
} Motti4SaplingStratum;

typedef struct
{
    Motti4SaplingStratum ma, ku, ra, hi, ha, hl, tl, mh, ml, _10;
} Motti4SaplingsSpev;

typedef Motti4SaplingsSpev Motti4Saplings[10];

typedef struct
{
    float year, V0, N0, alr, _5, type, amount, p, phos, _10;
} Motti4Fertilization;

typedef float Motti4VcrArray[270];
typedef float Motti4KorArray[2160];
typedef Motti4Fertilization Motti4FerArray[10];

typedef struct
{
    int death_forest, death_tree, _3, _4, _5, _6, _7, _8, calibrate, _10;
} Motti4Ctrl;

typedef struct
{
    float age100, h100, g, f, dg, spe;
} Motti4StoreyInfo;

typedef struct
{
    float Y, X, Z, lake, sea, dd;
    float _7, _8, _9, _10;
    float rgn_nat_spe, rgn_seedratio, rgn_vlj_spe, rgn_f, rgn_surv, _16;
    float xt_regen, xt_muok, xt_raiv, xt_fert_prev;
    float mal, mty, verl, verlt, alr, pd, _27, muok, _29, _30;
    Motti4Spev9 si;
    float tkg;
    Motti4Spev9 hd50;
    float year, step, sid, _53, xt_perk, prt, fthin, xt_thin, xt_fert, fert_peat;
    float xt_thoit, drain, xt_ndrain, xt_rdrain, _64, _65, _66, _67, _68;
    float xt_kar, xt_fthin, agedom, agedom13, ndom, spedom, spedom2, dcond, kehl, nstorey, gstorey;
    Motti4StoreyInfo st1, st2, st3, st4, st12;
    Motti4Spev10 hdom100, hdom_j, hg, hf, ddom100, ddom_latv, dg, df;
    float h100_perk, crdom, crerror, rimp, vg, v1, v2, v3, v4, v12;
    Motti4Spev10 f, f13, G, ccf;
    float _240[10];
    Motti4Spev10 ccfi, V;
    float f_dead, ba_dead, v_dead, _273, _274, _275, _276, _277, jh, jd;
    Motti4Spev10 xhdom;
    float _290, ddomg0, dgdom0, ba0, h100_0, cr100_0, v0, dg0, _298, dgM, _300;
    float _yy2[301];
} Motti4Site;

void Motti4SiteInit(Motti4Site* yy, float* Y, float* X, float* Z, int* rv);
void Motti4CheckYY(Motti4Site* yy, int* nerr, int* err);

void Motti4Init(Motti4Strata* yo,
                Motti4Site* yy,
                Motti4Saplings* ut,
                Motti4KorArray* kor,
                Motti4VcrArray* vcr,
                Motti4KorArray* apv,
                Motti4Trees* yp,
                Motti4Ctrl* o,
                int* numtrees,
                int* err,
                int* rv);

void Motti4InitVer2(Motti4Strata* yo,
                    Motti4Site* yy,
                    Motti4Saplings* ut,
                    Motti4KorArray* kor,
                    Motti4VcrArray* vcr,
                    Motti4KorArray* apv,
                    Motti4Trees* yp,
                    Motti4Ctrl* o,
                    int* numtrees,
                    int* err,
                    int* rv);

void Motti4UpdateAfterImport(Motti4Site* yy,
                             Motti4Trees* yp,
                             Motti4Saplings* ut,
                             Motti4KorArray* kor,
                             Motti4VcrArray* vcr,
                             Motti4KorArray* apv,
                             int* numtrees,
                             int* rv);

void Motti4Growth(Motti4Site* yy,
                  Motti4Trees* yp,
                  Motti4Saplings* ut,
                  Motti4KorArray* kor,
                  Motti4VcrArray* vcr,
                  Motti4KorArray* apv,
                  int* numtrees,
                  Motti4FerArray* fer,
                  int* numfer,
                  Motti4Ctrl* o,
                  int* step,
                  int* rv);

void Motti4Regenerate(float* method,
                      Motti4Site* yy,
                      Motti4Trees* yp,
                      Motti4Saplings* ut,
                      Motti4KorArray* kor,
                      Motti4VcrArray* vcr,
                      Motti4KorArray* apv,
                      int* numtrees,
                      int* step,
                      int* rv);

void Motti4PCTGuidelines(Motti4Site* yy,
                         Motti4Trees* yp,
                         Motti4Saplings* ut,
                         Motti4KorArray* kor,
                         Motti4VcrArray* vcr,
                         Motti4KorArray* apv,
                         int* numtrees,
                         int* remaingN,
                         int* rv);

void Motti4PCT(Motti4Site* yy,
               Motti4Trees* yp,
               Motti4Saplings* ut,
               Motti4KorArray* kor,
               Motti4VcrArray* vcr,
               Motti4KorArray* apv,
               int* numtrees,
               int* remaingN,
               int* rv);

void Motti4EarlyCare(Motti4Site* yy,
                     Motti4Trees* yp,
                     Motti4Saplings* ut,
                     Motti4KorArray* kor,
                     Motti4VcrArray* vcr,
                     Motti4KorArray* apv,
                     int* numtrees,
                     float* info,
                     int* imode,
                     int* rv);

void Motti4FillinPlanting(Motti4Site* yy,
                          Motti4Trees* yp,
                          Motti4Saplings* ut,
                          Motti4KorArray* kor,
                          Motti4VcrArray* vcr,
                          Motti4KorArray* apv,
                          int* numtrees,
                          int* rspe,
                          float* num,
                          int* ositeID,
                          int* rv);

void Motti4AfterSeedtreeCutting(Motti4Site* yy,
                                Motti4Trees* yp,
                                Motti4Saplings* ut,
                                Motti4KorArray* kor,
                                Motti4VcrArray* vcr,
                                Motti4KorArray* apv,
                                int* numtrees,
                                int* ierror,
                                int* rv);

void Motti4SeedingAgeShift(Motti4Site* yy, Motti4Saplings* ut, int* istep, int* rv);

void Motti4MineralSoilsFertilization(int* ftype,
                                     float* amountN,
                                     int* boolPhosporus,
                                     Motti4Site* yy,
                                     Motti4Trees* yp,
                                     Motti4Saplings* ut,
                                     Motti4KorArray* kor,
                                     Motti4VcrArray* vcr,
                                     Motti4KorArray* apv,
                                     int* numtrees,
                                     Motti4FerArray* fer,
                                     int* numfer,
                                     int* rv);

/* Optional helpers (best-effort) */
double Convert_Tree_Spec(double Mela_tree_spec_in);
float Convert_Site(int Mela_site);
void Pack_Tree_Matrix(void);
