#!/usr/bin/env python3
"""Table 4, clinician rows: three independent surgeon estimates.

Reviewer 1 (major comment 11) asked for a surgeon-prediction comparator. Three
spine surgeons who took no part in image generation, failure screening or
selection estimated each patient's postoperative major curve from the same
preoperative radiograph and the same structured plan that were supplied to the
model, recorded to 0.1 degrees, and were scored against the same blinded
reference measurement used throughout the manuscript.

Usage:
    python surgeon_comparison.py [--data PATH_TO_case_level_tables.xlsx]
                                 [--surgeons PATH_TO_surgeon_estimation_case_level.csv]

The surgeon table is aligned row-for-row with T1_Patient_Raw_357. That is
asserted rather than assumed: the preoperative major curve, the reference
postoperative major curve, the Lenke type and the planned upper instrumented
vertebra must agree for every patient before any statistic is computed.

Outputs
    Per surgeon, and for their three-way arithmetic mean: MAE, RMSE, threshold
    attainment at +/-5 and +/-10 degrees, and ICC(2,1) against the reference.
    Paired comparisons of each surgeon against the AI best-of-three workflow,
    by Wilcoxon signed-rank on the absolute errors and by exact McNemar on
    +/-5 degree attainment, with the 95% confidence interval of the paired mean
    difference.
    Agreement among the three surgeons: pairwise Pearson r and the mean absolute
    pairwise difference.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from agreement import icc21, wilcoxon_signed_rank
from checks import report, check, section, finish

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, os.pardir, 'data', 'case_level_tables.xlsx')
DEFAULT_SURGEONS = os.path.join(
    HERE, os.pardir, 'data', 'surgeon_estimation_case_level.csv')

# the column names in the surgeon table; remapped here so the script states its
# own expectations rather than depending on a particular delivery format
COL_ERROR = '真实术后主弯(°)'
SURGEONS = [('E1', 'E1', 4.57, 5.76, 60.7, 93.5, 0.749, 'attending, 8 years'),
            ('E2', 'E2', 4.47, 5.53, 60.5, 94.9, 0.766, 'attending, 11 years'),
            ('E3', 'E3', 4.30, 5.34, 65.0, 95.5, 0.776, 'associate chief, 16 years')]


def paired_mean_ci(a, b, alpha=0.05):
    """95% confidence interval of mean(a - b) from the paired t distribution."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(1 - alpha / 2, n - 1) * se
    return d.mean(), d.mean() - half, d.mean() + half


def mcnemar_exact(flag_a, flag_b):
    """Exact McNemar on two paired boolean arrays; returns (b, c, p)."""
    b = int((flag_a & ~flag_b).sum())
    c = int((~flag_a & flag_b).sum())
    if b + c == 0:
        return b, c, 1.0
    return b, c, stats.binomtest(b, b + c, 0.5).pvalue


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default=DEFAULT_DATA)
    ap.add_argument('--surgeons', default=DEFAULT_SURGEONS)
    args = ap.parse_args()
    for path in (args.data, args.surgeons):
        if not os.path.exists(path):
            sys.exit(f"input not found: {path}")

    print(f"cohort  : {args.data}")
    print(f"surgeons: {args.surgeons}")
    print(f"python  : {sys.version.split()[0]}")

    df = pd.read_excel(args.data, sheet_name='T1_Patient_Raw_357')
    sg = pd.read_csv(args.surgeons, encoding='utf-8-sig')

    section('Join check: the surgeon table must align with the cohort')
    check('same number of patients', len(df) == len(sg), f'{len(df)} vs {len(sg)}')
    check('preoperative curve agrees',
          np.allclose(df.pre_cobb_main.values, sg['术前主弯(°)'].values, atol=1e-9))
    check('reference postoperative curve agrees',
          np.allclose(df.post_cobb_main.values, sg[COL_ERROR].values, atol=1e-9))
    check('Lenke type agrees',
          bool((df.lenke_type.values == sg['Lenke分型'].values).all()))
    check('planned upper instrumented vertebra agrees',
          bool((df.uiv.values == sg['计划UIV'].values).all()))

    actual = df.post_cobb_main.values
    ai = df.ai_cobb_main.values
    panel = sg[['E1', 'E2', 'E3']].mean(axis=1).values

    section('Surgeon estimates against the blinded reference (N = %d)' % len(df))
    rows = {}
    for key, col, mae, rmse, w5, w10, icc, who in SURGEONS:
        pred = sg[col].values
        err = pred - actual
        rows[key] = pred
        print(f"\n  {key} ({who})")
        report(f'{key}: MAE (deg)', round(np.abs(err).mean(), 2), mae, '{:.2f}')
        report(f'{key}: RMSE (deg)', round(np.sqrt((err ** 2).mean()), 2), rmse, '{:.2f}')
        report(f'{key}: within +/-5 deg (%)', round(100 * (np.abs(err) <= 5).mean(), 1),
               w5, '{:.1f}')
        report(f'{key}: within +/-10 deg (%)', round(100 * (np.abs(err) <= 10).mean(), 1),
               w10, '{:.1f}')
        report(f'{key}: ICC(2,1)', round(icc21(actual, pred), 3), icc, '{:.3f}')

    print('\n  panel mean of the three surgeons')
    perr = panel - actual
    report('panel mean: MAE (deg)', round(np.abs(perr).mean(), 2), 4.22, '{:.2f}')
    report('panel mean: RMSE (deg)', round(np.sqrt((perr ** 2).mean()), 2), 5.25, '{:.2f}')
    report('panel mean: within +/-5 deg (%)', round(100 * (np.abs(perr) <= 5).mean(), 1),
           63.3, '{:.1f}')
    report('panel mean: within +/-10 deg (%)',
           round(100 * (np.abs(perr) <= 10).mean(), 1), 95.5, '{:.1f}')
    report('panel mean: ICC(2,1)', round(icc21(actual, panel), 3), 0.783, '{:.3f}')

    section('AI best-of-three against each surgeon (paired, N = %d)' % len(df))
    d_ai = np.abs(ai - actual)
    report('AI: MAE (deg)', round(d_ai.mean(), 2), 3.43, '{:.2f}')
    report('AI: within +/-5 deg (%)', round(100 * (d_ai <= 5).mean(), 1), 76.3, '{:.1f}')
    diffs = []
    for key, col, *_rest in SURGEONS:
        d_s = np.abs(sg[col].values - actual)
        m, lo, hi = paired_mean_ci(d_s, d_ai)
        diffs.append(m)
        _z, p = wilcoxon_signed_rank(d_ai, d_s)
        b, c, pm = mcnemar_exact(d_ai <= 5, d_s <= 5)
        print(f"\n  AI vs {key}")
        print(f"      paired MAE difference {m:+.2f} deg, 95% CI [{lo:+.2f}, {hi:+.2f}]")
        check(f'{key}: confidence interval excludes zero', lo > 0, f'[{lo:+.2f}, {hi:+.2f}]')
        check(f'{key}: AI lower MAE', d_ai.mean() < d_s.mean(),
              f'{d_ai.mean():.2f} vs {d_s.mean():.2f} deg')
        print(f"      Wilcoxon signed-rank p = {p:.3g}")
        print(f"      exact McNemar, +/-5 deg attainment: b = {b}, c = {c}, p = {pm:.3g}")
        check(f'{key}: AI higher +/-5 deg attainment', (d_ai <= 5).mean() > (d_s <= 5).mean(),
              f'{100 * (d_ai <= 5).mean():.1f}% vs {100 * (d_s <= 5).mean():.1f}%')

    m, lo, hi = paired_mean_ci(np.abs(panel - actual), d_ai)
    print("\n  AI vs the three-surgeon panel mean")
    print(f"      paired MAE difference {m:+.2f} deg, 95% CI [{lo:+.2f}, {hi:+.2f}]")
    report('AI advantage over the panel mean (deg)', round(m, 2), 0.79, '{:.2f}')
    report('  lower 95% confidence limit', round(lo, 2), 0.37, '{:.2f}')
    report('  upper 95% confidence limit', round(hi, 2), 1.21, '{:.2f}')
    report('AI advantage, smallest of the four (deg)', round(min(diffs + [m]), 2), 0.79, '{:.2f}')
    report('AI advantage, largest of the four (deg)', round(max(diffs + [m]), 2), 1.14, '{:.2f}')

    section('Agreement among the three surgeons')
    rs, mads = [], []
    for a, b in [('E1', 'E2'), ('E1', 'E3'), ('E2', 'E3')]:
        r, _ = stats.pearsonr(sg[a].values, sg[b].values)
        mad = np.abs(sg[a].values - sg[b].values).mean()
        rs.append(r)
        mads.append(mad)
        print(f"  {a} vs {b}: Pearson r = {r:.3f}; mean absolute difference = {mad:.2f} deg")
    report('pairwise r, lowest', round(min(rs), 3), 0.937, '{:.3f}')
    report('pairwise r, highest', round(max(rs), 3), 0.940, '{:.3f}')
    report('mean absolute pairwise difference, lowest (deg)',
           round(min(mads), 2), 2.47, '{:.2f}')
    report('mean absolute pairwise difference, highest (deg)',
           round(max(mads), 2), 2.48, '{:.2f}')

    finish('the manuscript')


if __name__ == '__main__':
    main()
