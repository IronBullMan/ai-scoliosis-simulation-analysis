"""Figure 5 -- retrospectively perceived potential communication utility, B convention.
Regenerated without the baked-in title and without the publisher watermark."""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Patch
import figstyle
from figstyle import *

_ap = argparse.ArgumentParser(description='Regenerate this figure.')
figstyle.add_data_arg(_ap)
_ap.add_argument('--outdir', default='.', metavar='DIR', help='output directory')
_args = _ap.parse_args()
figstyle.set_data(_args.data)

A, B, ids = load()
post, ai, pre, corr, d = core(A, B, ids)
CLS = {k: ('acceptable' if abs(v) <= 5 else ('over' if v < -5 else 'under'))
       for k, v in zip(ids, d)}
X = pd.ExcelFile(figstyle.CASE)
T3A = pd.read_excel(X, 'T3A_Patient_Survey'); T3A['cls'] = T3A.patient_id.map(CLS)
T3B = pd.read_excel(X, 'T3B_Clinician_Raw'); T3B['cls'] = T3B.patient_id.map(CLS)
T3B['d_B'] = T3B.patient_id.map(dict(zip(ids, d)))
ORD = ['acceptable', 'under', 'over']
COL = {'acceptable': GREEN, 'under': BLUE, 'over': RED}
LBL = {'acceptable': 'Acceptable\n(|\u0394| \u2264 5\u00b0)', 'under': 'Under-simulation',
       'over': 'Over-simulation'}

fig, axes = plt.subplots(2, 2, figsize=(13.42, 11.72))

# ---------------------------------------------------------------- A
ax = axes[0, 0]
ad = np.abs(T3B.d_B.values); cu = T3B.comm_utility_1to5.values
ax.scatter(ad, cu, s=7, color=SCAT, alpha=0.28, linewidths=0, zorder=2)
edges = np.arange(0, 10.5, 0.5)
ctr, mu, se = [], [], []
for i in range(len(edges) - 1):
    m = (ad >= edges[i]) & (ad < edges[i + 1])
    if m.sum() >= 8:
        ctr.append((edges[i] + edges[i + 1]) / 2); mu.append(cu[m].mean())
        se.append(cu[m].std(ddof=1) / np.sqrt(m.sum()))
ctr, mu, se = map(np.array, (ctr, mu, se))
ax.fill_between(ctr, mu - se, mu + se, color=RED, alpha=0.18, zorder=3)
ax.plot(ctr, mu, '-o', color=RED, lw=2.2, ms=6, zorder=4, label='Binned mean \u00b1 SE')
ax.axvline(5, ls='--', color='#444444', lw=1.4, zorder=3)
rho, pv = stats.spearmanr(cu, ad)
ax.text(0.03, 0.965, f'Spearman \u03c1 = {rho:.3f}\np {"< 0.001" if pv < 1e-3 else f"= {pv:.4f}"}',
        transform=ax.transAxes, va='top', fontsize=9.5,
        bbox=dict(fc='white', ec='#bbbbbb', boxstyle='round,pad=0.35'))
ax.set_xlabel('|\u0394Cobb| (AI \u2212 Actual, degrees)')
ax.set_ylabel('Clinician-Rated Communication Utility (1\u20135)')
ax.set_title('A   Communication Utility vs. |\u0394Cobb|', fontdict=SANS, loc='left', pad=8)
ax.set_ylim(0.9, 5.35); ax.set_xlim(-0.2, 10.4)
ax.legend(loc='upper right', fontsize=9)
despine(ax)

# ---------------------------------------------------------------- B
ax = axes[0, 1]
g = [T3A.loc[T3A.cls == k, 'concordance_1to5'].values for k in ORD]
vp = ax.violinplot(g, positions=range(3), widths=0.8, showextrema=False, showmedians=False)
for body, k in zip(vp['bodies'], ORD):
    body.set_facecolor(COL[k]); body.set_alpha(0.55); body.set_edgecolor(COL[k])
for i, (v, k) in enumerate(zip(g, ORD)):
    ax.scatter(np.random.default_rng(7).normal(i, 0.045, len(v)), v, s=11,
               color=COL[k], alpha=0.75, linewidths=0, zorder=4)
    ax.hlines(np.median(v), i - 0.22, i + 0.22, color='black', lw=2.2, zorder=5)
    ax.hlines(v.mean(), i - 0.22, i + 0.22, color=RED, lw=2.2, zorder=5)
    ax.text(i, max(np.median(v), v.mean()) + 0.14, f'{v.mean():.2f}', ha='center',
            va='bottom', fontsize=10.5, zorder=6)
p1 = stats.mannwhitneyu(g[0], g[1]).pvalue
p2 = stats.mannwhitneyu(g[0], g[2]).pvalue
kw = stats.kruskal(*g).pvalue
ax.plot([0, 0, 1, 1], [4.75, 4.85, 4.85, 4.75], color='#555555', lw=1.2)
ax.text(0.5, 4.88, 'p < 0.001' if p1 < 1e-3 else f'p = {p1:.4f}', ha='center', fontsize=9.5)
ax.plot([0, 0, 2, 2], [5.15, 5.25, 5.25, 5.15], color='#555555', lw=1.2)
ax.text(1.0, 5.28, f'p = {p2:.4f}', ha='center', fontsize=9.5)
ax.set_xticks(range(3))
ax.set_xticklabels([f'{LBL[k].splitlines()[0]}\n(n={len(g[i])})' for i, k in enumerate(ORD)],
                   fontsize=9.5)
ax.set_ylabel('Perceived Concordance (1\u20135)')
ax.set_title('B   Patient Perceived Concordance\n      (Kruskal-Wallis p '
             + ('< 0.001)' if kw < 1e-3 else f'= {kw:.4f})'), fontdict=SANS, loc='left', pad=8)
ax.set_ylim(0.9, 5.6); ax.set_xlim(-0.55, 2.55)
despine(ax)

# ---------------------------------------------------------------- C
ax = axes[1, 0]
gc = [T3B.loc[T3B.cls == k, 'comm_utility_1to5'].values for k in ORD]
bp = ax.boxplot(gc, positions=range(3), widths=0.5, patch_artist=True, showfliers=True,
                medianprops=dict(color=RED, lw=2.2),
                flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='#555555',
                                markersize=5))
for patch, k in zip(bp['boxes'], ORD):
    patch.set_facecolor(COL[k]); patch.set_alpha(0.45); patch.set_edgecolor('#555555')
for i, (v, k) in enumerate(zip(gc, ORD)):
    ax.scatter(np.random.default_rng(11).normal(i, 0.055, len(v)), v, s=3.2,
               color=COL[k], alpha=0.32, linewidths=0, zorder=3)
    ax.text(i, v.mean() + 0.10, f'{v.mean():.2f}', ha='center', fontsize=10.5, zorder=6)
    ax.text(i, 1.045, f'n={len(v)}', ha='center', fontsize=8.5, color='#555555')
F = stats.f_oneway(*gc)
ax.set_xticks(range(3)); ax.set_xticklabels([LBL[k] for k in ORD], fontsize=9.5)
ax.set_ylabel('Clinician-Rated Communication Utility (1\u20135)')
ax.set_title('C   Clinician Communication Utility by Error Category\n'
             f'      (One-way ANOVA F = {F.statistic:.1f}, p < 0.001)',
             fontdict=SANS, loc='left', pad=8, fontsize=11.5)
ax.set_ylim(0.9, 5.35); ax.set_xlim(-0.55, 2.55)
despine(ax)

# ---------------------------------------------------------------- D
ax = axes[1, 1]
cli = [100 * T3B.loc[T3B.cls == k, 'would_use_for_counseling'].mean() for k in ORD]
pat = [100 * T3A.loc[T3A.cls == k, 'would_recommend'].mean() for k in ORD]
x = np.arange(3); w = 0.34
b1 = ax.bar(x - w / 2, cli, w, color=NAVY, label='Clinician Would-Use', zorder=3)
b2 = ax.bar(x + w / 2, pat, w, color=LIGHTBLUE, edgecolor=NAVY, hatch='///', linewidth=0,
            label='Patient Would-Recommend', zorder=3)
for xi, v in zip(x - w / 2, cli):
    ax.text(xi, v + 1.5, f'{v:.0f}%', ha='center', fontsize=10, color=NAVY)
for xi, v in zip(x + w / 2, pat):
    ax.text(xi, v + 1.5, f'{v:.0f}%', ha='center', fontsize=10, color='#2a6fb0')
ax.set_xticks(x); ax.set_xticklabels([LBL[k] for k in ORD], fontsize=9.5)
ax.set_ylabel('Rate (%)'); ax.set_ylim(0, 108)
ax.set_title('D   Willingness to Use / Recommend by Error Category', fontdict=SANS,
             loc='left', pad=8)
ax.legend(loc='upper right', fontsize=9.5)
despine(ax)
ax.set_axisbelow(True)

plt.tight_layout(pad=1.6, h_pad=2.6)
plt.savefig(os.path.join(_args.outdir, 'Figure5_utility.png'), bbox_inches='tight', facecolor='white')
print('saved Figure5_utility.png')
