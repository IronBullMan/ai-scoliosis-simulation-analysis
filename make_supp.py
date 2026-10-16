"""Supplementary Figures S1-S5, regenerated under the B convention."""
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
S = stats_json(); EX = S['extra']
CLS = {k: ('acceptable' if abs(v) <= 5 else ('over' if v < -5 else 'under'))
       for k, v in zip(ids, d)}
X = pd.ExcelFile(figstyle.CASE)
T3A = pd.read_excel(X, 'T3A_Patient_Survey'); T3A['cls'] = T3A.patient_id.map(CLS)
T3B = pd.read_excel(X, 'T3B_Clinician_Raw'); T3B['cls'] = T3B.patient_id.map(CLS)
ORD = ['acceptable', 'under', 'over']
COL = {'acceptable': GREEN, 'under': BLUE, 'over': RED}
LBL2 = {'acceptable': 'Acceptable\n(|\u0394| \u2264 5\u00b0)', 'under': 'Under-simulation',
        'over': 'Over-simulation'}

# =================================================================== S1
rows = [('Major Cobb Angle (\u00b0)\n(Primary Endpoint)', S['tier1']['icc'],
         S['tier1']['icc_ci'], S['tier1']['r'], True),
        ('C7-CSVL (mm)', S['coronal']['c7_csvl_mm']['icc'], S['coronal']['c7_csvl_mm']['icc_ci'],
         S['coronal']['c7_csvl_mm']['r'], False),
        ('T1 Tilt (\u00b0)', S['coronal']['t1_tilt_deg']['icc'], S['coronal']['t1_tilt_deg']['icc_ci'],
         S['coronal']['t1_tilt_deg']['r'], False),
        ('Clavicle Angle (\u00b0)', S['coronal']['clavicle_deg']['icc'],
         S['coronal']['clavicle_deg']['icc_ci'], S['coronal']['clavicle_deg']['r'], False)]

fig, ax = plt.subplots(figsize=(17.8, 8.8))
LABX = max(r[2][1] for r in rows) + 0.008
for i, (lab, icc, ci, r, primary) in enumerate(rows):
    y = len(rows) - 1 - i
    c = NAVY if primary else MID
    rcol = '#9fb3c8' if primary else '#7fb8e8'
    ax.errorbar(icc, y, xerr=[[icc - ci[0]], [ci[1] - icc]], fmt='s', ms=13, color=c,
                ecolor=c, elinewidth=2, capsize=5, capthick=1.6, zorder=4)
    ax.plot(r, y, marker='D', ms=11, color=rcol, mec=c if primary else rcol, mew=1.4, zorder=5)
    ax.text(r, y + 0.17, f'r={r:.3f}', color=c, fontsize=12, ha='center')
    ax.text(LABX, y, f'ICC={icc:.3f}', color=c, fontsize=13, va='center', ha='left')
ax.axvline(0.75, ls='--', color='#3aa76d', lw=1.6, zorder=1)
ax.axvline(0.90, ls=':', color=BLUE, lw=1.6, zorder=1)
ax.text(0.75, 3.72, 'Good\n(0.75)', color='#2e8b57', fontsize=11.5, ha='center', va='bottom')
ax.text(0.90, 3.72, 'Excellent\n(0.90)', color='#1a5fa8', fontsize=11.5, ha='center', va='bottom')
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=13)
ax.set_xlim(0.70, 0.96); ax.set_ylim(-0.55, 3.95)
ax.set_xticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
ax.set_xlabel('ICC(2,1) / Pearson r', fontsize=14)
ax.set_title('Structural Agreement Metrics (Forest Plot)', fontdict=SERIF, fontsize=17, pad=14)
ax.legend(handles=[plt.Line2D([], [], marker='s', ls='', color=NAVY, ms=12,
                              label='ICC(2,1) [95% CI]'),
                   plt.Line2D([], [], marker='D', ls='', color='#9fb3c8', ms=10,
                              label="Pearson's r")],
          loc='lower right', fontsize=11.5)
ax.grid(False)
ax.set_axisbelow(True)
despine(ax)
plt.tight_layout(); plt.savefig(os.path.join(_args.outdir, 'FigureS1_forest.png'), facecolor='white'); plt.close()
print('S1 done')

# =================================================================== S2
LK = EX['S2']; types = ['1', '2', '3', '4', '5', '6']
ns = [LK[t]['n'] for t in types]
fig, axes = plt.subplots(1, 2, figsize=(25.0, 9.3))
ax = axes[0]
x = np.arange(6); w = 0.38
ax.bar(x - w / 2, [LK[t]['w3'] for t in types], w, color=LIGHT, label='Within \u00b13\u00b0 (Strict)')
ax.bar(x + w / 2, [LK[t]['w5'] for t in types], w, color=NAVY, label='Within \u00b15\u00b0 (Clinical)')
for xi, t in zip(x + w / 2, types):
    ax.text(xi, LK[t]['w5'] + 1.5, f'{LK[t]["w5"]:.0f}%', ha='center', fontsize=11)
ax.set_xticks(x); ax.set_xticklabels([f'Type {t}\n(n={n})' for t, n in zip(types, ns)], fontsize=11)
ax.set_ylabel('Cases (%)'); ax.set_ylim(0, 105)
ax.set_title('A   Within-Threshold Rate by Lenke Type', fontdict=SERIF, fontsize=15, loc='left')
ax.legend(loc='lower left', fontsize=11)
despine(ax); ax.set_axisbelow(True)
ax = axes[1]
ax2 = ax.twinx()
ax.bar(x, [LK[t]['mae'] for t in types], 0.5, color=ORANGE, label='MAE (\u00b0)')
for xi, t in zip(x, types):
    ax.text(xi, LK[t]['mae'] - 0.45, f'{LK[t]["mae"]:.1f}\u00b0', ha='center', va='top',
            color='white', fontsize=11.5, fontweight='bold')
ax2.plot(x, [LK[t]['r'] for t in types], '-o', color=NAVY, lw=2.6, ms=9)
for xi, t in zip(x, types):
    ax2.text(xi, LK[t]['r'] + 0.008, f'{LK[t]["r"]:.3f}', ha='center', color=NAVY, fontsize=11)
ax.set_xticks(x); ax.set_xticklabels([f'Type {t}\n(n={n})' for t, n in zip(types, ns)], fontsize=11)
ax.set_ylabel('MAE (\u00b0)'); ax.set_ylim(0, 7); ax2.set_ylim(0.70, 0.97)
ax2.set_ylabel("Pearson's r"); ax2.grid(False)
ax.set_title('B   MAE and Pearson r by Lenke Type', fontdict=SERIF, fontsize=15, loc='left')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=11)
despine(ax); despine(ax2, keep=('right',)); ax2.spines['right'].set_visible(True)
ax.set_axisbelow(True); ax2.set_axisbelow(True)
plt.tight_layout(); plt.savefig(os.path.join(_args.outdir, 'FigureS2_lenke.png'), facecolor='white'); plt.close()
print('S2 done')

# =================================================================== S3
fig, axes = plt.subplots(1, 2, figsize=(25.0, 9.4))
for ax, xv, lab, col, rk, pk, xlab in [
        (axes[0], pre, 'A   Absolute Error vs. Preoperative Curve Severity', PINK,
         EX['S3']['rho_pre'], EX['S3']['p_pre'], 'Preoperative Major Cobb Angle (\u00b0)'),
        (axes[1], corr, 'B   Absolute Error vs. Correction Rate', NAVY,
         EX['S3']['rho_corr'], EX['S3']['p_corr'], 'Surgical Correction Rate (%)')]:
    ax.scatter(xv, np.abs(d), s=13, color=SCAT, alpha=0.28, linewidths=0, zorder=2)
    e = np.linspace(xv.min(), xv.max(), 9)
    ctr, mu = [], []
    for i in range(len(e) - 1):
        m = (xv >= e[i]) & (xv < e[i + 1])
        if m.sum() >= 5:
            ctr.append((e[i] + e[i + 1]) / 2); mu.append(np.abs(d)[m].mean())
    ax.plot(ctr, mu, '-o', color=col, lw=2.6, ms=9, zorder=4, label='Binned mean')
    ax.axhline(5, ls='--', color='#666666', lw=1.6, zorder=3)
    ax.text(0.02, 0.965, f'Spearman \u03c1 = {rk:.3f}, p = {pk:.4f}', transform=ax.transAxes,
            va='top', fontsize=12.5,
            bbox=dict(fc='white', ec='#888888', boxstyle='round,pad=0.3'))
    ax.set_xlabel(xlab, fontsize=13); ax.set_ylabel('|\u0394Cobb| (AI \u2212 Actual, \u00b0)', fontsize=13)
    ax.set_ylim(-0.4, 10.8)
    ax.set_title(lab, fontdict=SERIF, fontsize=15, loc='left')
    ax.legend(loc='center right', fontsize=11)
    despine(ax); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(os.path.join(_args.outdir, 'FigureS3_correlates.png'), facecolor='white'); plt.close()
print('S3 done')

# =================================================================== S4
S4 = EX['S4']
fig, axes = plt.subplots(2, 2, figsize=(23.0, 17.9))
for ax, (lab, sub) in zip(axes.ravel(), S4.items()):
    x = np.arange(3)
    ax.bar(x, [sub['mean'][k] for k in ORD], 0.55, color=[COL[k] for k in ORD],
           yerr=[sub['sd'][k] for k in ORD], capsize=5,
           error_kw=dict(ecolor='#444444', lw=1.3))
    for xi, k in zip(x, ORD):
        ax.text(xi, sub['mean'][k] + sub['sd'][k] + 0.09, f'{sub["mean"][k]:.2f}',
                ha='center', fontsize=18)
        ax.text(xi, 1.07, f'n={sub["n"][k]}', ha='center', fontsize=12, color='#666666',
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))
    p = sub['kw_p']
    ax.set_title(f'{lab}\n(Kruskal-Wallis p {"< 0.001" if p < 1e-3 else f"= {p:.4f}"})',
                 fontdict=SERIF, fontsize=16, pad=14)
    ax.set_xticks(x); ax.set_xticklabels([LBL2[k] for k in ORD], fontsize=12)
    ax.set_ylabel('Score (1\u20135)', fontsize=13); ax.set_ylim(1, 5.5)
    despine(ax); ax.set_axisbelow(True)
fig.suptitle('Tier 3A \u2014 Patient Retrospective Perception Survey: All Dimensions',
             fontsize=19, y=0.985, family='DejaVu Serif', weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.965]); plt.savefig(os.path.join(_args.outdir, 'FigureS4_survey.png'), facecolor='white')
plt.close(); print('S4 done')

# =================================================================== S5
fig, axes = plt.subplots(1, 2, figsize=(24.4, 9.2))
ax = axes[0]
rat = ['Clinician A (Mid-career)', 'Clinician B (Senior)', 'Clinician C (Senior)']
rc = [NAVY, '#3f9ae0', '#4fc7a0']
x = np.arange(3); w = 0.26
for j, (r_, c_) in enumerate(zip(rat, rc)):
    vals = [100 * T3B.loc[(T3B.rater == r_) & (T3B.cls == k), 'would_use_for_counseling'].mean()
            for k in ORD]
    ax.bar(x + (j - 1) * w, vals, w, color=c_, label=r_)
    for xi, v in zip(x + (j - 1) * w, vals):
        ax.text(xi, v + 1.6, f'{v:.0f}%', ha='center', fontsize=11.5)
ax.set_xticks(x)
ax.set_xticklabels(['Acceptable\n(|\u0394| \u2264 5\u00b0)', 'Under-simulation\n(\u0394 > +5\u00b0)',
                    'Over-simulation\n(\u0394 < \u22125\u00b0)'], fontsize=12)
ax.set_ylabel('Would Use for Counselling (%)', fontsize=13); ax.set_ylim(0, 108)
ax.set_title('A   Would-Use Rate by Clinician and Error Category', fontdict=SERIF, fontsize=15,
             loc='left')
ax.legend(loc='upper center', fontsize=11)
despine(ax); ax.set_axisbelow(True)
ax = axes[1]
vals = [T3B.loc[T3B.rater == r_, 'comm_utility_1to5'].values for r_ in rat]
bx = ax.boxplot(vals, positions=range(3), widths=0.5, patch_artist=True,
                medianprops=dict(color=RED, lw=2.4),
                flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='#555555',
                                markersize=6))
for patch, c_ in zip(bx['boxes'], rc):
    patch.set_facecolor(c_); patch.set_alpha(0.5); patch.set_edgecolor('#555555')
for i, v in enumerate(vals):
    ax.scatter(np.random.default_rng(3).normal(i, 0.05, len(v)), v, s=3.5, color='#444444',
               alpha=0.3, linewidths=0, zorder=3)
ax.set_xticks(range(3))
ax.set_xticklabels([f'{r_}\n({v.mean():.2f} \u00b1 {v.std(ddof=1):.2f})'
                    for r_, v in zip(rat, vals)], fontsize=11)
ax.set_ylabel('Communication Utility (1\u20135)', fontsize=13); ax.set_ylim(0.9, 5.3)
ax.set_title('B   Communication Utility Distribution by Clinician', fontdict=SERIF, fontsize=15,
             loc='left')
despine(ax); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(os.path.join(_args.outdir, 'FigureS5_clinician.png'), facecolor='white'); plt.close()
print('S5 done')
