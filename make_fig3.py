"""Figure 3 -- structural agreement. Regenerated under the B convention."""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
import figstyle
from figstyle import *

_ap = argparse.ArgumentParser(description='Regenerate this figure.')
figstyle.add_data_arg(_ap)
_ap.add_argument('--outdir', default='.', metavar='DIR', help='output directory')
_args = _ap.parse_args()
figstyle.set_data(_args.data)

A, B, ids = load()
post, ai, pre, corr, d = core(A, B, ids)
S = stats_json()['tier1']

fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.85))
ax = axes[0]
ax.scatter(post, ai, s=13, color=SCAT, alpha=0.5, linewidths=0, zorder=2)
lim = [min(post.min(), ai.min()) - 2, max(post.max(), ai.max()) + 2]
ax.plot(lim, lim, ls='--', lw=2, color=PINK, zorder=3, label='Identity (y = x)')
sl, ic = np.polyfit(post, ai, 1)
xs = np.array([post.min(), post.max()])
ax.plot(xs, sl * xs + ic, lw=2, color='#283850', zorder=4,
        label=f'Fit (r = {S["r"]:.3f})')
ax.axvline(0, lw=0); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Actual Postoperative Cobb Angle (\u00b0)')
ax.set_ylabel('AI-Simulated Cobb Angle (\u00b0)')
ax.set_title('A   Cobb Angle: AI Simulation vs. Actual', fontdict=SERIF, pad=8)
ax.text(0.04, 0.965, f'Within \u00b15\u00b0 = {S["within5_pct"]:.0f}%', transform=ax.transAxes,
        va='top', fontsize=10.5)
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, loc='upper left', bbox_to_anchor=(0.02, 0.85), fontsize=9)
despine(ax)

ax = axes[1]
mean = (post + ai) / 2
ax.axhspan(S['loa'][0], S['loa'][1], color='#f0f0f0', zorder=1)
ax.scatter(mean, d, s=13, color=SCAT, alpha=0.5, linewidths=0, zorder=2)
ax.axhline(S['loa'][1], ls='--', lw=1.8, color='#444444', zorder=3,
           label=f'95% LoA [{S["loa"][0]:.1f}, {S["loa"][1]:.1f}]\u00b0')
ax.axhline(S['loa'][0], ls='--', lw=1.8, color='#444444', zorder=3)
ax.axhline(S['bias'], lw=2.2, color=RED, zorder=4, label=f'Bias = {S["bias"]:.2f}\u00b0')
ax.axhline(5, ls=':', lw=1.4, color=ORANGE, zorder=3)
ax.axhline(-5, ls=':', lw=1.4, color=ORANGE, zorder=3)
ax.text(ax.get_xlim()[1], 5.15, '+5\u00b0', color='#c98a2e', fontsize=9, ha='right', va='bottom')
ax.text(ax.get_xlim()[1], -5.15, '\u22125\u00b0', color='#c98a2e', fontsize=9, ha='right', va='top')
ax.set_xlabel('Mean Cobb Angle (Actual + AI) / 2 (\u00b0)')
ax.set_ylabel('\u0394Cobb (AI \u2212 Actual) (\u00b0)')
ax.set_title('B   Bland-Altman Analysis', fontdict=SERIF, pad=8)
h, l = ax.get_legend_handles_labels()
ax.legend([h[1], h[0]], [l[1], l[0]], loc='upper right', fontsize=9)
despine(ax)

ax = axes[2]
bins = np.arange(np.floor(d.min()), np.ceil(d.max()) + 1, 1.0)
for lo, hi, col, lab in [(-1e9, -5, RED, None), (5, 1e9, BLUE, None), (-5, 5, GREEN, None)]:
    sel = d[(d > lo) & (d <= hi)] if lab is None else None
    ax.hist(d[(d > lo) & (d <= hi)], bins=bins, color=col, alpha=0.75,
            edgecolor='white', linewidth=0.6, zorder=2)
ax.axvline(0, color='black', lw=1.6, zorder=4)
ax.axvline(-5, color=RED, ls='--', lw=1.6, zorder=4)
ax.axvline(5, color=BLUE, ls='--', lw=1.6, zorder=4)
ax.set_xlabel('\u0394Cobb (AI \u2212 Actual) (\u00b0)')
ax.set_ylabel('Number of Patients')
ax.set_title(f'C   Distribution of \u0394Cobb (N = {S["n"]})', fontdict=SERIF, pad=8)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor=GREEN, alpha=0.75,
          label=f'Acceptable (|\u0394| \u2264 5\u00b0): {S["within5"]} ({S["within5_pct"]:.0f}%)'),
    Patch(facecolor=RED, alpha=0.75,
          label=f'Over-simulation (\u0394 < \u22125\u00b0): {S["over"]} ({S["over_pct"]:.0f}%)'),
    Patch(facecolor=BLUE, alpha=0.75,
          label=f'Under-simulation (\u0394 > +5\u00b0): {S["under"]} ({S["under_pct"]:.0f}%)')],
    loc='upper left', fontsize=9)
despine(ax)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(_args.outdir, 'Figure3_structural_agreement.png'), bbox_inches='tight', facecolor='white')
print('saved Figure3_structural_agreement.png')
