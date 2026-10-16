"""
Compute every statistic the figures need, under the B convention
(reference standard = mean of the two blinded rater measurements for the actual
postoperative Cobb angle; the simulated Cobb angle is the value reported in the main
patient table, which is unchanged by the convention).

Output: figstats.json (override with --out)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

R3 = lambda x, n=3: None if x is None else round(float(x), n)
OUT = {}

_ap = argparse.ArgumentParser(description='Recompute every statistic the figures plot.')
_ap.add_argument('--data', default=os.path.join(os.pardir, 'data', 'case_level_tables.xlsx'),
                 metavar='PATH', help='case-level table (default: %(default)s)')
_ap.add_argument('--out', default='figstats.json', metavar='PATH',
                 help='where to write the computed values (default: %(default)s)')
_args = _ap.parse_args()
CASE = _args.data

# Single input table: the reference standard is the mean of the two blinded rater
# measurements, so every column is read from the same workbook.
X = pd.ExcelFile(CASE)
A = pd.read_excel(CASE, 'T1_Patient_Raw_357').set_index('patient_id')
ids = list(A.index)
post = A.loc[ids, 'post_cobb_main'].values.astype(float)       # mean of two blinded raters
ai = A.loc[ids, 'ai_cobb_main'].values.astype(float)
pre = A.loc[ids, 'pre_cobb_main'].values.astype(float)
corr = A.loc[ids, 'corr_rate_pct'].values.astype(float)
d = ai - post


def wilson(k, n, z=1.959963985):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * max(0, c - h), 100 * min(1, c + h))


def icc21_ci(x, y):
    m = np.column_stack([x, y]).astype(float)
    n, k = m.shape
    g = m.mean()
    ssr = k * ((m.mean(1) - g) ** 2).sum()
    ssc = n * ((m.mean(0) - g) ** 2).sum()
    sse = ((m - g) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    v = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    F = msr / mse
    lo = (F / stats.f.ppf(0.975, n - 1, (n - 1) * (k - 1)) - 1) / \
         (F / stats.f.ppf(0.975, n - 1, (n - 1) * (k - 1)) + k - 1)
    hi = (F / stats.f.ppf(0.025, n - 1, (n - 1) * (k - 1)) - 1) / \
         (F / stats.f.ppf(0.025, n - 1, (n - 1) * (k - 1)) + k - 1)
    return float(v), float(lo), float(hi)


def ccc_ci(x, y, n_boot=10000, seed=42):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    sxy = ((x - mx) * (y - my)).mean()
    val = 2 * sxy / (vx + vy + (mx - my) ** 2)
    rng = np.random.default_rng(seed)
    n = len(x)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        a, b = x[i], y[i]
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(), b.var()
        sab = ((a - ma) * (b - mb)).mean()
        bs.append(2 * sab / (va + vb + (ma - mb) ** 2))
    return float(val), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ------------------------------------------------------------------ Tier 1 (figure 3)
bias, sd = d.mean(), d.std(ddof=0)
lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
r = float(np.corrcoef(ai, post)[0, 1])
icc, icc_lo, icc_hi = icc21_ci(post, ai)
ccc, ccc_lo, ccc_hi = ccc_ci(post, ai)
OUT['tier1'] = dict(
    n=len(d), post_mean=R3(post.mean(), 1), post_sd=R3(post.std(ddof=1), 1),
    ai_mean=R3(ai.mean(), 1), ai_sd=R3(ai.std(ddof=1), 1),
    bias=R3(bias, 2), loa=[R3(lo, 1), R3(hi, 1)],
    mae=R3(np.abs(d).mean(), 2), rmse=R3(np.sqrt((d ** 2).mean()), 2),
    r=R3(r), icc=R3(icc), icc_ci=[R3(icc_lo), R3(icc_hi)],
    ccc=R3(ccc), ccc_ci=[R3(ccc_lo), R3(ccc_hi)],
    within3=int((np.abs(d) <= 3).sum()), within3_pct=R3(100 * (np.abs(d) <= 3).mean(), 1),
    within5=int((np.abs(d) <= 5).sum()), within5_pct=R3(100 * (np.abs(d) <= 5).mean(), 1),
    within10=int((np.abs(d) <= 10).sum()), within10_pct=R3(100 * (np.abs(d) <= 10).mean(), 1),
    over=int((d < -5).sum()), over_pct=R3(100 * (d < -5).mean(), 1),
    under=int((d > 5).sum()), under_pct=R3(100 * (d > 5).mean(), 1),
)

# ------------------------------------------------------------------ S1 forest plot: secondary coronal parameters
cor = {}
for name, act, sim in [('c7_csvl_mm', 'c7_csvl_actual_mm', 'c7_csvl_ai_mm'),
                       ('t1_tilt_deg', 't1_tilt_actual_deg', 't1_tilt_ai_deg'),
                       ('clavicle_deg', 'clavicle_actual_deg', 'clavicle_ai_deg')]:
    x = A.loc[ids, act].values.astype(float)
    y = A.loc[ids, sim].values.astype(float)
    v, l, h = icc21_ci(x, y)
    cor[name] = dict(icc=R3(v), icc_ci=[R3(l), R3(h)],
                     r=R3(float(np.corrcoef(x, y)[0, 1])),
                     actual_mean=R3(x.mean(), 1), actual_sd=R3(x.std(ddof=1), 1),
                     ai_mean=R3(y.mean(), 1), ai_sd=R3(y.std(ddof=1), 1),
                     delta_mean=R3((y - x).mean(), 1), delta_sd=R3((y - x).std(ddof=1), 1))
OUT['coronal'] = cor

# ------------------------------------------------------------------ S2 Lenke subgroups
lenke = {}
for t in sorted(A['lenke_type'].unique()):
    m = A.loc[ids, 'lenke_type'].values == t
    dd = d[m]
    lenke[str(int(t))] = dict(
        n=int(m.sum()),
        within3_pct=R3(100 * (np.abs(dd) <= 3).mean(), 1),
        within5_pct=R3(100 * (np.abs(dd) <= 5).mean(), 1),
        mae=R3(np.abs(dd).mean(), 2), rmse=R3(np.sqrt((dd ** 2).mean()), 2),
        r=R3(float(np.corrcoef(ai[m], post[m])[0, 1])))
OUT['lenke'] = lenke

# ------------------------------------------------------------------ S3 error correlates
OUT['correlates'] = dict(
    rho_pre=R3(float(stats.spearmanr(np.abs(d), pre).statistic)),
    p_pre=R3(float(stats.spearmanr(np.abs(d), pre).pvalue), 4),
    rho_corr=R3(float(stats.spearmanr(np.abs(d), corr).statistic)),
    p_corr=R3(float(stats.spearmanr(np.abs(d), corr).pvalue), 4),
)

# ------------------------------------------------------------------ figure 4: Tier 2
T2 = pd.read_excel(X, 'T2_Turing_Raw_120')
raters = ['r1', 'r2', 'r3', 'r4']
years = {'r1': 7, 'r2': 9, 'r3': 18, 'r4': 22}
group = {'r1': 'mid', 'r2': 'mid', 'r3': 'sen', 'r4': 'sen'}
t2 = {}
for r_ in raters:
    k = int(T2[f'{r_}_correct'].sum())
    n = len(T2)
    tp = int(((T2[f'{r_}_correct'] == 1) & (T2.is_ai == 1)).sum())
    tn = int(((T2[f'{r_}_correct'] == 1) & (T2.is_ai == 0)).sum())
    fp = int(((T2[f'{r_}_correct'] == 0) & (T2.is_ai == 0)).sum())
    fn = int(((T2[f'{r_}_correct'] == 0) & (T2.is_ai == 1)).sum())
    sens = 100 * tp / (tp + fn)
    spec = 100 * tn / (tn + fp)
    f1 = 2 * tp / (2 * tp + fp + fn)
    t2[r_] = dict(years=years[r_], group=group[r_], n=n, acc=100 * k / n,
                  ci=[R3(v, 1) for v in wilson(k, n)],
                  p=R3(float(stats.binomtest(k, n, 0.5).pvalue), 4),
                  sens=R3(sens, 1), spec=R3(spec, 1), f1=R3(f1),
                  realism_ai=R3(T2.loc[T2.is_ai == 1, f'{r_}_realism'].mean(), 1),
                  realism_real=R3(T2.loc[T2.is_ai == 0, f'{r_}_realism'].mean(), 1))
mid = [t2[r_]['acc'] for r_ in raters if group[r_] == 'mid']
sen = [t2[r_]['acc'] for r_ in raters if group[r_] == 'sen']
allc = [t2[r_]['acc'] for r_ in raters]
OUT['tier2'] = dict(per_rater=t2, overall=R3(np.mean(allc), 1),
                    mid_pooled=R3(np.mean(mid), 1), sen_pooled=R3(np.mean(sen), 1))

# ------------------------------------------------------------------ B-convention error class per patient
cls = np.where(np.abs(d) <= 5, 'acceptable', np.where(d < -5, 'over', 'under'))
clsmap = dict(zip(ids, cls))
OUT['class_counts'] = {k: int((cls == k).sum()) for k in ('acceptable', 'under', 'over')}

# ------------------------------------------------------------------ Tier 3A (figure 5B, S4)
T3A = pd.read_excel(X, 'T3A_Patient_Survey')
T3A['cls'] = T3A.patient_id.map(clsmap)
dims = ['concordance_1to5', 'understanding_retro_1to5', 'confidence_retro_1to5',
        'satisfaction_retro_1to5']
t3a = {'n': int(len(T3A))}
for dim in dims:
    g = {}
    for k in ('acceptable', 'under', 'over'):
        v = T3A.loc[T3A.cls == k, dim].values
        g[k] = dict(n=int(len(v)), mean=R3(v.mean(), 2), sd=R3(v.std(ddof=1), 2))
    allv = T3A[dim].values
    kw = stats.kruskal(*[T3A.loc[T3A.cls == k, dim].values for k in ('acceptable', 'under', 'over')])
    t3a[dim] = dict(groups=g, all_mean=R3(allv.mean(), 2), all_sd=R3(allv.std(ddof=1), 2),
                    kw_p=R3(float(kw.pvalue), 4))
    wr = {}
    for k in ('acceptable', 'under', 'over'):
        v = T3A.loc[T3A.cls == k, 'would_recommend'].values
        wr[k] = dict(n=int(len(v)), pct=R3(100 * v.mean(), 1))
    t3a[dim + '_recommend'] = wr
OUT['tier3a'] = t3a

# ------------------------------------------------------------------ Tier 3B (figure 5C,D, S5)
T3B = pd.read_excel(X, 'T3B_Clinician_Raw')
T3B['cls'] = T3B.patient_id.map(clsmap)
t3b = {}
for k in ('acceptable', 'under', 'over'):
    s = T3B[T3B.cls == k]
    t3b[k] = dict(n=int(len(s)), utility_mean=R3(s.comm_utility_1to5.mean(), 2),
                  utility_sd=R3(s.comm_utility_1to5.std(ddof=1), 2),
                  use_pct=R3(100 * s.would_use_for_counseling.mean(), 1))
grp = [T3B.loc[T3B.cls == k, 'comm_utility_1to5'].values for k in ('acceptable', 'under', 'over')]
t3b['anova_F'] = R3(float(stats.f_oneway(*grp).statistic), 1)
t3b['anova_p'] = R3(float(stats.f_oneway(*grp).pvalue), 4)
per = {}
for r_ in sorted(T3B.rater.unique()):
    s = T3B[T3B.rater == r_]
    per[r_] = dict(
        utility_mean=R3(s.comm_utility_1to5.mean(), 2),
        utility_sd=R3(s.comm_utility_1to5.std(ddof=1), 2),
        use_pct=R3(100 * s.would_use_for_counseling.mean(), 1),
        by_class={k: dict(n=int(len(s[s.cls == k])),
                          use_pct=R3(100 * s.loc[s.cls == k, 'would_use_for_counseling'].mean(), 1))
                  for k in ('acceptable', 'under', 'over')})
t3b['per_rater'] = per
OUT['tier3b'] = t3b
OUT['tier3b_utility_vs_error'] = dict(
    rho=R3(float(stats.spearmanr(T3B.comm_utility_1to5, np.abs(T3B.delta_cobb_main)).statistic)),
    p=R3(float(stats.spearmanr(T3B.comm_utility_1to5, np.abs(T3B.delta_cobb_main)).pvalue), 4))

json.dump(OUT, open(_args.out, 'w'), indent=1, ensure_ascii=False)
print(json.dumps(OUT, indent=1, ensure_ascii=False))

# ==================================================================== additions
extra = {}
# per-rating B delta for the Tier 3 tables (patient and clinician rows carry the A delta column)
T3A['d_B'] = T3A.patient_id.map(dict(zip(ids, d)))
T3B['d_B'] = T3B.patient_id.map(dict(zip(ids, d)))

# figure 5 panel A: clinician utility vs |delta_B|
r_, p_ = stats.spearmanr(T3B.comm_utility_1to5, np.abs(T3B.d_B))
extra['fig5A'] = dict(rho=R3(float(r_)), p=R3(float(p_), 4))
# patient concordance vs |delta_B|
r2_, p2_ = stats.spearmanr(T3A.concordance_1to5, np.abs(T3A.d_B))
extra['fig5B_rho'] = dict(rho=R3(float(r2_)), p=R3(float(p2_), 4))

# figure 5 panel B: violin groups + Kruskal-Wallis + pairwise Mann-Whitney (acceptable vs over)
gB = {k: T3A.loc[T3A.cls == k, 'concordance_1to5'].values for k in ('acceptable', 'under', 'over')}
extra['fig5B'] = dict(
    n={k: int(len(v)) for k, v in gB.items()},
    mean={k: R3(v.mean(), 2) for k, v in gB.items()},
    sd={k: R3(v.std(ddof=1), 2) for k, v in gB.items()},
    kw_p=R3(float(stats.kruskal(*gB.values()).pvalue), 4),
    p_acc_vs_over=R3(float(stats.mannwhitneyu(gB['acceptable'], gB['over']).pvalue), 4))
# figure 5 panel C: box groups + ANOVA
gC = {k: T3B.loc[T3B.cls == k, 'comm_utility_1to5'].values for k in ('acceptable', 'under', 'over')}
extra['fig5C'] = dict(
    n={k: int(len(v)) for k, v in gC.items()},
    mean={k: R3(v.mean(), 2) for k, v in gC.items()},
    sd={k: R3(v.std(ddof=1), 2) for k, v in gC.items()},
    F=R3(float(stats.f_oneway(*gC.values()).statistic), 1),
    p=R3(float(stats.f_oneway(*gC.values()).pvalue), 6))
# figure 5 panel D: willingness
extra['fig5D'] = dict(
    clinician={k: R3(100 * T3B.loc[T3B.cls == k, 'would_use_for_counseling'].mean(), 1)
               for k in ('acceptable', 'under', 'over')},
    patient={k: R3(100 * T3A.loc[T3A.cls == k, 'would_recommend'].mean(), 1)
             for k in ('acceptable', 'under', 'over')})
# S4: all four dimensions with exact KW p
s4 = {}
for dim, lab in [('concordance_1to5', 'Concordance'), ('understanding_retro_1to5', 'Understanding'),
                 ('confidence_retro_1to5', 'Confidence'), ('satisfaction_retro_1to5', 'Satisfaction')]:
    g = {k: T3A.loc[T3A.cls == k, dim].values for k in ('acceptable', 'under', 'over')}
    s4[lab] = dict(n={k: int(len(v)) for k, v in g.items()},
                   mean={k: R3(v.mean(), 2) for k, v in g.items()},
                   sd={k: R3(v.std(ddof=1), 2) for k, v in g.items()},
                   kw_p=float(stats.kruskal(*g.values()).pvalue))
extra['S4'] = s4
extra['S3'] = dict(rho_pre=R3(float(stats.spearmanr(np.abs(d), pre).statistic)),
                   p_pre=R3(float(stats.spearmanr(np.abs(d), pre).pvalue), 4),
                   rho_corr=R3(float(stats.spearmanr(np.abs(d), corr).statistic)),
                   p_corr=R3(float(stats.spearmanr(np.abs(d), corr).pvalue), 4))
extra['S2'] = {str(t): dict(n=int((A.loc[ids, 'lenke_type'].values == int(t)).sum()),
                       w3=R3(100 * (np.abs(d[A.loc[ids, 'lenke_type'].values == int(t)]) <= 3).mean(), 1),
                       w5=R3(100 * (np.abs(d[A.loc[ids, 'lenke_type'].values == int(t)]) <= 5).mean(), 1),
                       mae=R3(np.abs(d[A.loc[ids, 'lenke_type'].values == int(t)]).mean(), 2),
                       r=R3(float(np.corrcoef(ai[A.loc[ids, 'lenke_type'].values == int(t)],
                                              post[A.loc[ids, 'lenke_type'].values == int(t)])[0, 1])))
               for t in sorted(A['lenke_type'].unique())}
OUT['extra'] = extra
json.dump(OUT, open(_args.out, 'w'), indent=1, ensure_ascii=False)
print(json.dumps(extra, indent=1, ensure_ascii=False))
