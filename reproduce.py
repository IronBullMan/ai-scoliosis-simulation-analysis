#!/usr/bin/env python3
"""Reproduce the statistics reported in the manuscript.

Usage:
    python reproduce.py [--data PATH_TO_case_level_tables.xlsx] [--ablation PATH]

The script prints one line per reported quantity: the value it computed, and the
value printed in the manuscript. Differences are collected and summarised at the
end, so a mismatch cannot pass unnoticed.

Inputs
    case_level_tables.xlsx  case-level analysis tables for the primary study
    --ablation           the prompt-sensitivity workbook (optional); the
                         corresponding section is skipped if it is absent
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from agreement import (icc21, icc21_ci, ccc, ccc_ci, bland_altman, wilson,
                       pearson_ci, cohens_kappa, wilcoxon_signed_rank)
from checks import PROBLEMS, report, section, finish

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, os.pardir, 'data', 'case_level_tables.xlsx')
DEFAULT_ABLATION = os.path.join(HERE, os.pardir, 'data', 'ablation_record.xlsx')


# --------------------------------------------------------------------------
# Tier 1 -- structural agreement between simulated and actual Cobb angles
# --------------------------------------------------------------------------
def tier1(df):
    section('Tier 1: structural agreement (N = %d)' % len(df))
    actual, sim = df.post_cobb_main.values, df.ai_cobb_main.values
    d = sim - actual

    report('patients', len(df), 354, '{:d}')
    report('actual postoperative Cobb, mean (deg)', round(actual.mean(), 1), 15.9, '{:.1f}')
    report('actual postoperative Cobb, SD (deg)', round(actual.std(ddof=1), 1), 7.0, '{:.1f}')
    report('AI-simulated Cobb, mean (deg)', round(sim.mean(), 1), 16.6, '{:.1f}')
    report('AI-simulated Cobb, SD (deg)', round(sim.std(ddof=1), 1), 7.9, '{:.1f}')
    report('delta Cobb, mean (deg)', round(d.mean(), 2), 0.72, '{:.2f}')
    report('delta Cobb, SD (deg)', round(d.std(ddof=1), 2), 4.05, '{:.2f}')
    report('mean absolute error (deg)', round(np.abs(d).mean(), 2), 3.26, '{:.2f}')
    report('root mean square error (deg)', round(np.sqrt((d ** 2).mean()), 2), 4.11, '{:.2f}')
    for thr, want_n, want_pct, want_lo, want_hi in [(3, 193, 54.5, 49.3, 59.6),
                                                    (5, 289, 81.6, 77.3, 85.3),
                                                    (10, 345, 97.5, 95.2, 98.7)]:
        k = int((np.abs(d) <= thr).sum())
        lo, hi = wilson(k, len(d))
        report(f'within +/-{thr} deg (n)', k, want_n, '{:d}')
        report(f'within +/-{thr} deg (%)', round(100 * k / len(d), 1), want_pct, '{:.1f}')
        report(f'within +/-{thr} deg 95% CI low (%)', round(100 * lo, 1), want_lo, '{:.1f}')
        report(f'within +/-{thr} deg 95% CI high (%)', round(100 * hi, 1), want_hi, '{:.1f}')
    report('over-simulation, d < -5 deg (%)', round(100 * (d < -5).mean(), 1), 5.4, '{:.1f}')
    report('over-simulation (n)', int((d < -5).sum()), 19, '{:d}')
    report('under-simulation, d > +5 deg (%)', round(100 * (d > 5).mean(), 1), 13.0, '{:.1f}')
    report('under-simulation (n)', int((d > 5).sum()), 46, '{:d}')
    v, vlo, vhi = icc21_ci(actual, sim)
    report('ICC(2,1)', round(v, 3), 0.850, '{:.3f}')
    report('ICC(2,1) 95% CI low', round(vlo, 3), 0.823, '{:.3f}')
    report('ICC(2,1) 95% CI high', round(vhi, 3), 0.880, '{:.3f}')
    v, vlo, vhi = ccc_ci(actual, sim)
    report('Lin CCC', round(v, 3), 0.850, '{:.3f}')
    report('Lin CCC 95% CI low', round(vlo, 3), 0.819, '{:.3f}')
    report('Lin CCC 95% CI high', round(vhi, 3), 0.875, '{:.3f}')
    r, rci = pearson_ci(actual, sim)
    report('Pearson r', round(r, 2), 0.86, '{:.2f}')
    ba = bland_altman(actual, sim)
    report('Bland-Altman bias (deg)', round(ba['bias'], 2), 0.72, '{:.2f}')
    report('Bland-Altman bias 95% CI low', round(ba['bias_ci'][0], 2), 0.30, '{:.2f}')
    report('Bland-Altman bias 95% CI high', round(ba['bias_ci'][1], 2), 1.14, '{:.2f}')
    report('limits of agreement low (deg)', round(ba['loa'][0], 1), -7.2, '{:.1f}')
    report('limits of agreement high (deg)', round(ba['loa'][1], 1), 8.7, '{:.1f}')

    section('Tier 1: secondary coronal-balance parameters')
    for label, ca, ci, want_icc, want_r in [
            ('C7-CSVL distance (mm)', 'c7_csvl_actual_mm', 'c7_csvl_ai_mm', 0.879, 0.893),
            ('T1 tilt (deg)', 't1_tilt_actual_deg', 't1_tilt_ai_deg', 0.872, 0.885),
            ('clavicle angle (deg)', 'clavicle_actual_deg', 'clavicle_ai_deg', 0.891, 0.901)]:
        a, s = df[ca].values, df[ci].values
        report(f'{label} ICC(2,1)', round(icc21(a, s), 3), want_icc, '{:.3f}')
        rr, _ = pearson_ci(a, s)
        report(f'{label} Pearson r', round(rr, 3), want_r, '{:.3f}')


def measurement_reliability(path):
    section('Measurement reliability (two readers)')
    md = pd.read_excel(path, sheet_name='T1d_双测量者Cobb')
    report('postoperative radiographs ICC(2,1)', round(icc21(md['术后_测量者1'], md['术后_测量者2']), 2),
           0.93, '{:.2f}')
    report('AI simulations ICC(2,1)', round(icc21(md['AI_测量者1'], md['AI_测量者2']), 2), 0.91, '{:.2f}')


# --------------------------------------------------------------------------
# Generation workflow and failure screening
# --------------------------------------------------------------------------
FAILURE_LABELS = {
    '解剖畸变（骨性解剖严重扭曲）': 'anatomical distortion',
    '视觉上不合理的矫正不足': 'visually implausible undercorrection',
    '脊柱序列不真实': 'unrealistic spinal alignment',
    '植入物幻觉（植入物幻觉）': 'implant hallucination',
    '视觉上不合理的过度矫正': 'visually implausible overcorrection',
    '图像质量失败（未生成或无法判读内固定）': 'image-quality failure',
    '非诊断性输出（未生成或无法判读内固定）': 'non-diagnostic output',
    '左右/方向错误（左右方向错误）': 'laterality or orientation error',
}


def failures(path):
    section('Generation attempts and failure screening')
    g = pd.read_excel(path, sheet_name='T1b_逐次生成记录')
    failed = g[g['判定'] == '失败']
    report('generation attempts', len(g), 1062, '{:d}')
    report('failed generations', len(failed), 159, '{:d}')
    rate = len(failed) / len(g)
    report('failure rate (%)', round(100 * rate, 1), 15.0, '{:.1f}')
    lo, hi = wilson(len(failed), len(g))
    report('failure rate 95% CI low (%)', round(100 * lo, 1), 13.0, '{:.1f}')
    report('failure rate 95% CI high (%)', round(100 * hi, 1), 17.2, '{:.1f}')

    counts = failed['失败类型'].value_counts()
    for cn, en in FAILURE_LABELS.items():
        report('  ' + en, int(counts.get(cn, 0)), EXPECTED_FAILURE[en], '{:d}')

    # the two prompt versions, and whether they differ in failure rate
    v3 = g[g['提示词版本'] == 'final_v3']
    v4 = g[g['提示词版本'] == 'final_v4']
    section('Template revision: first 40 patients vs the remaining 314')
    report('generations, earlier template', len(v3), 120, '{:d}')
    report('generations, final template', len(v4), 942, '{:d}')
    tab = np.array([[int((v3['判定'] == '失败').sum()), int((v3['判定'] == '通过').sum())],
                    [int((v4['判定'] == '失败').sum()), int((v4['判定'] == '通过').sum())]])
    odds, p = stats.fisher_exact(tab)
    print(f"      failure counts {tab[:, 0].tolist()}; Fisher exact p = {p:.4g}")


EXPECTED_FAILURE = {
    'anatomical distortion': 39,
    'visually implausible undercorrection': 34,
    'unrealistic spinal alignment': 31,
    'implant hallucination': 19,
    'visually implausible overcorrection': 15,
    'image-quality failure': 11,
    'non-diagnostic output': 7,
    'laterality or orientation error': 3,
}


# --------------------------------------------------------------------------
# The human-in-the-loop workflow
# --------------------------------------------------------------------------
def human_in_the_loop(path):
    section('Human-in-the-loop workflow')
    per = pd.read_excel(path, sheet_name='T1b2_逐例汇总')
    sel = pd.read_excel(path, sheet_name='T1c_选择记录')
    gen = pd.read_excel(path, sheet_name='T1b_逐次生成记录')

    n_fail = per['失败数'].value_counts().to_dict()
    report('patients with 0 failed attempts', int(n_fail.get(0, 0)), 211, '{:d}')
    report('patients with 1 failed attempt', int(n_fail.get(1, 0)), 127, '{:d}')
    report('patients with 2 failed attempts', int(n_fail.get(2, 0)), 16, '{:d}')
    report('patients with 3 failed attempts', int(n_fail.get(3, 0)), 0, '{:d}')
    report('patients with >=1 acceptable candidate', int((per['成功候选数'] >= 1).sum()), 354, '{:d}')

    chosen = sel['选择者1_选中代数'].value_counts().to_dict()
    for attempt, want in [(1, 112), (2, 116), (3, 126)]:
        report(f'primary selection = attempt {attempt}', int(chosen.get(attempt, 0)), want, '{:d}')

    # the spread statistics use the patients in whom all three attempts yielded a
    # measurable residual curve, whether or not the attempt passed the failure screen
    measurable = gen.dropna(subset=['|误差|(°)']).groupby('病例号').size()
    allthree = per[per['病例号'].isin(measurable[measurable == 3].index)]
    report('patients with all three attempts measurable', len(allthree), 343, '{:d}')
    report('median within-patient SD (deg)', round(allthree['三次SD(°)'].median(), 2), 3.16, '{:.2f}')
    report('median best-minus-worst (deg)', round(allthree['最好-最差(°)'].median(), 2), 5.91, '{:.2f}')

    first = per['首张误差(°)'].values
    picked = per['选用图误差(°)'].values
    report('first output within +/-5 deg (%)', round(100 * (first <= 5).mean(), 1), 68.0, '{:.1f}', tol=0.3)
    report('selected output within +/-5 deg (%)', round(100 * (picked <= 5).mean(), 1), 81.6, '{:.1f}')
    report('MAE, selected output (deg)', round(picked.mean(), 2), 3.26, '{:.2f}')
    report('MAE, first output (deg)', round(first.mean(), 2), 3.95, '{:.2f}')

    # the manuscript reports this paired difference with a Wilcoxon signed-rank
    # p-value; check the inequality it states rather than a point estimate
    z, p = wilcoxon_signed_rank(picked, first)
    ok = p < 0.001
    if not ok:
        PROBLEMS.append(('Wilcoxon p, selected vs first output', p, '< 0.001'))
    print(f"  {'ok ' if ok else 'DIFF'} {'Wilcoxon p, selected vs first':<46} "
          f"computed {p:<22.4g} reported < 0.001")
    r, _ = stats.pearsonr(picked, first)
    print(f"      (normal approximation, z = {z:.3f}; scipy exact would give "
          f"p = {stats.wilcoxon(picked, first).pvalue:.4g})")

    # geometric optimum: the best candidate each patient could have had
    best = gen.groupby('病例号')['|误差|(°)'].min()
    report('geometric optimum within +/-5 deg (%)', round(100 * (best <= 5).mean(), 1), 83.6, '{:.1f}')

    section('Reproducibility of the selection step')
    s1, s2, s3 = (sel['选择者1_选中代数'], sel['选择者2_选中代数'], sel['选择者3_选中代数'])
    for label, a, b, want_n, want_pct, want_k in [
            ('selector 1 vs 2', s1, s2, 232, 65.5, 0.482),
            ('selector 1 vs 3', s1, s3, 202, 57.1, 0.356),
            ('selector 2 vs 3', s2, s3, 156, 44.1, 0.161)]:
        agree = int((a == b).sum())
        report(f'{label}: agreement (n)', agree, want_n, '{:d}')
        report(f'{label}: agreement (%)', round(100 * agree / len(sel), 1), want_pct, '{:.1f}')
        report(f'{label}: Cohen kappa', round(cohens_kappa(a, b), 3), want_k, '{:.3f}')
    allsame = int(((s1 == s2) & (s2 == s3)).sum())
    report('all three selectors agreed (n)', allsame, 132, '{:d}')
    report('all three selectors agreed (%)', round(100 * allsame / len(sel), 1), 37.3, '{:.1f}')

    print('\n  error of each selector\'s own choice')
    err = gen.set_index(['病例号', '代数'])['|误差|(°)']
    for label, col, want_mae, want_pct, want_n in [
            ('primary selector', '选择者1_选中代数', 3.26, 81.6, 354),
            ('mid-career re-selector', '选择者2_选中代数', 3.59, 76.1, 351),
            ('senior re-selector', '选择者3_选中代数', 3.77, 72.0, 353)]:
        key = list(zip(sel['病例号'], sel[col]))
        vals = err.reindex(key).values.astype(float)
        vals = vals[~np.isnan(vals)]
        report(f'{label}: n with a measurable curve', len(vals), want_n, '{:d}')
        report(f'{label}: MAE (deg)', round(np.abs(vals).mean(), 2), want_mae, '{:.2f}')
        report(f'{label}: within +/-5 deg (%)', round(100 * (np.abs(vals) <= 5).mean(), 1), want_pct, '{:.1f}')


# --------------------------------------------------------------------------
# Tier 2 -- forced-choice visual plausibility
# --------------------------------------------------------------------------
def tier2(path):
    section('Tier 2: visual plausibility (forced choice)')
    t = pd.read_excel(path, sheet_name='T2_Turing_Raw_120')
    ai = t['is_ai'].values == 1
    for k, label, want_acc, want_sens, want_spec, want_p in [
            (1, 'R1 (mid-career, 7 yr)', 49.2, 50.0, 48.3, 0.9273),
            (2, 'R2 (mid-career, 9 yr)', 51.7, 50.0, 53.3, 0.7843),
            (3, 'R3 (senior, 18 yr)', 61.7, 63.3, 60.0, 0.0134),
            (4, 'R4 (senior, 22 yr)', 64.2, 60.0, 68.3, 0.0024)]:
        correct = t[f'r{k}_correct'].values == 1
        n = len(t)
        acc = 100 * correct.mean()
        sens = 100 * correct[ai].mean()      # correctly calling a simulation a simulation
        spec = 100 * correct[~ai].mean()     # correctly calling a real radiograph real
        p = stats.binomtest(int(correct.sum()), n, 0.5).pvalue
        report(f'{label}: accuracy (%)', round(acc, 1), want_acc, '{:.1f}')
        report(f'{label}: sensitivity (%)', round(sens, 1), want_sens, '{:.1f}')
        report(f'{label}: specificity (%)', round(spec, 1), want_spec, '{:.1f}')
        report(f'{label}: p vs chance', round(p, 4), want_p, '{:.4f}')
    mid = [(t[f'r{k}_correct'].values == 1) for k in (1, 2)]
    sen = [(t[f'r{k}_correct'].values == 1) for k in (3, 4)]
    report('mid-career pooled accuracy (%)', round(100 * np.mean(mid), 1), 50.4, '{:.1f}')
    report('senior pooled accuracy (%)', round(100 * np.mean(sen), 1), 62.9, '{:.1f}')


# --------------------------------------------------------------------------
# Tier 3A -- patient survey
# --------------------------------------------------------------------------
def tier3a(path):
    section('Tier 3A: patient survey by simulation-error category')
    s = pd.read_excel(path, sheet_name='T3A_Patient_Survey')
    domains = [('concordance_1to5', 'AI-actual concordance', None),
               ('understanding_retro_1to5', 'retrospective understanding', None),
               ('confidence_retro_1to5', 'retrospective confidence', None),
               ('satisfaction_retro_1to5', 'satisfaction', None)]
    groups = [s[s.bias_class == c] for c in ['acceptable', 'under', 'over']]
    for col, label, _ in domains:
        h, p = stats.kruskal(*[g[col].values for g in groups])
        means = [round(g[col].mean(), 2) for g in groups]
        print(f"  {label:<30} means {means}  Kruskal-Wallis p = {p:.4g}")
    report('patients surveyed', len(s), 50, '{:d}')
    report('  acceptable / under / over',
           ' / '.join(str(int((s.bias_class == c).sum())) for c in ['acceptable', 'under', 'over']),
           '35 / 10 / 5')


# --------------------------------------------------------------------------
# Tier 3B -- clinician ratings
# --------------------------------------------------------------------------
def tier3b(path):
    section('Tier 3B: clinician ratings, mixed-effects and GEE')
    c = pd.read_excel(path, sheet_name='T3B_Clinician_Raw')
    c['absd'] = c.delta_cobb_main.abs()
    report('clinician-image ratings', len(c), 1062, '{:d}')
    report('raters', c.rater.nunique(), 3, '{:d}')
    report('mean communication utility', round(c.comm_utility_1to5.mean(), 2), 3.90, '{:.2f}')
    report('utility SD', round(c.comm_utility_1to5.std(ddof=1), 2), 0.53, '{:.2f}')
    report('would-use for counselling (%)', round(100 * c.would_use_for_counseling.mean(), 1), 79.0, '{:.1f}', tol=0.6)

    # inter-rater reliability of the utility rating, clinician treated as rater
    w = c.pivot_table(index='patient_id', columns='rater', values='comm_utility_1to5')
    m = w.values
    n, k = m.shape
    grand = m.mean()
    ss_r = k * ((m.mean(1) - grand) ** 2).sum()
    ss_c = n * ((m.mean(0) - grand) ** 2).sum()
    ss_e = ((m - grand) ** 2).sum() - ss_r - ss_c
    msr, msc, mse = ss_r / (n - 1), ss_c / (k - 1), ss_e / ((n - 1) * (k - 1))
    report('inter-rater ICC(2,1), utility', round((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n), 2),
           0.30, '{:.2f}')

    lmm = smf.mixedlm('comm_utility_1to5 ~ absd', c, groups=c['rater']).fit()
    report('mixed model: beta per degree', round(lmm.params['absd'], 3), -0.123, '{:.3f}')
    report('mixed model: p', '< 0.001' if lmm.pvalues['absd'] < 0.001 else round(lmm.pvalues['absd'], 4), '< 0.001')
    print(f"      random-intercept variance = {float(lmm.cov_re.iloc[0, 0]):.3g} "
          f"(3 clusters; the estimate is at the boundary, so the fit reduces to OLS)")

    gee = smf.gee('would_use_for_counseling ~ absd', groups='rater', data=c,
                  family=sm.families.Binomial()).fit()
    report('GEE: odds ratio per degree', round(np.exp(gee.params['absd']), 3), 0.795, '{:.3f}')
    report('GEE: p', '< 0.001' if gee.pvalues['absd'] < 0.001 else round(gee.pvalues['absd'], 4), '< 0.001')


# --------------------------------------------------------------------------
# Prompt sensitivity (ablation) -- optional
# --------------------------------------------------------------------------
def prompt_sensitivity(path):
    if not os.path.exists(path):
        print(f"\n=== Prompt sensitivity: SKIPPED ({path} not found) ===\n")
        return
    section('Prompt sensitivity (four prompt conditions)')
    per = pd.read_excel(path, sheet_name='逐例汇总')
    want = {'A': 55.0, 'B': 83.3, 'C': 68.3, 'D': 81.7}
    for arm in ['A', 'B', 'C', 'D']:
        sub = per[per['臂'] == arm]
        report(f'arm {arm}: cases within +/-5 deg (%)',
               round(100 * (sub['选用图达标'] == '是').mean(), 1), want[arm], '{:.1f}')
    a = (per[per['臂'] == 'A'].set_index('病例号')['选用图达标'] == '是')
    b = (per[per['臂'] == 'B'].set_index('病例号')['选用图达标'] == '是')
    common = a.index.intersection(b.index)
    tab = np.array([[int((a[common] & b[common]).sum()), int((a[common] & ~b[common]).sum())],
                    [int((~a[common] & b[common]).sum()), int((~a[common] & ~b[common]).sum())]])
    p = stats.binomtest(int(tab[0, 1]), int(tab[0, 1] + tab[1, 0]), 0.5).exact_pvalue if False else \
        stats.binomtest(int(tab[0, 1]), int(tab[0, 1] + tab[1, 0]), 0.5).pvalue
    print(f"      exact McNemar, complete vs concise prompt: p = {p:.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DEFAULT_DATA)
    ap.add_argument('--ablation', default=DEFAULT_ABLATION)
    args = ap.parse_args()
    if not os.path.exists(args.data):
        sys.exit(f"data file not found: {args.data}")

    print(f"data   : {args.data}")
    print(f"python : {sys.version.split()[0]}  numpy {np.__version__}  pandas {pd.__version__}")

    df = pd.read_excel(args.data, sheet_name='T1_Patient_Raw_357')
    tier1(df)
    measurement_reliability(args.data)
    failures(args.data)
    human_in_the_loop(args.data)
    tier2(args.data)
    tier3a(args.data)
    tier3b(args.data)
    prompt_sensitivity(args.ablation)

    finish('the manuscript')


if __name__ == '__main__':
    main()
