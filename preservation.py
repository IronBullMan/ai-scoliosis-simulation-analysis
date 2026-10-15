#!/usr/bin/env python3
"""Exploratory regional comparison of the simulations against the preoperative image.

For the 78 patients for whom a saved simulation and both the preoperative and the
real postoperative radiograph were available (one simulation per patient), each
simulation and each postoperative radiograph was affinely registered to that
patient's own preoperative radiograph by maximisation of mutual information, so
that only global translation, rotation, scaling and shear were permitted and no
local deformation could be imposed. Four regions were defined proportionally on
the preoperative image:

    pelvis             iliac wings and sacrum
    peri_lumbar        lower lumbar spine
    cervical_shoulder  cervicothoracic region including the shoulder girdle
    spine_field        instrumented thoracolumbar segment

Within each region three quantities were computed: the normalised cross-
correlation of grey values (`ncc_*`), the same correlation computed on gradient-
magnitude images (`gncc_*`), and the residual local displacement by block
matching (`disp_*`), the last expressed both in pixels and as a percentage of the
inter-iliac width.

Usage:
    python preservation.py [--metrics PATH_TO_preservation_metrics.csv]

The registration itself needs the image files and a registration library, and is
performed by a separate script that is not distributed here. This script starts
from the per-image metrics table that registration produced: one row per image,
two rows (AI simulation, real postoperative radiograph) per patient.

That metrics table is not distributed with this repository, both because it
depends on the images and because the delivery format carries patient
identifiers. Only the `target` column and the numeric metric columns are read
here; nothing in this script depends on the identifier column.
"""

import argparse
import os
import sys

import pandas as pd

from agreement import wilcoxon_signed_rank
from checks import report, check, section, finish

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_METRICS = os.path.join(HERE, os.pardir, 'preservation', 'preservation_metrics.csv')

AI, POSTOP = 'AI', 'POSTOP'
REGIONS = [('pelvis', 'iliac wings and sacrum'),
           ('peri_lumbar', 'lower lumbar spine'),
           ('cervical_shoulder', 'cervicothoracic region and shoulder girdle'),
           ('spine_field', 'instrumented thoracolumbar segment')]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--metrics', default=DEFAULT_METRICS)
    args = ap.parse_args()
    if not os.path.exists(args.metrics):
        sys.exit(f"metrics file not found: {args.metrics}\n"
                 "The registration step that produces it needs the image files "
                 "and is not distributed with this repository.")

    print(f"metrics: {args.metrics}")
    print(f"python : {sys.version.split()[0]}  pandas {pd.__version__}")
    d = pd.read_csv(args.metrics, encoding='utf-8-sig')
    ai = d[d.target == AI]
    po = d[d.target == POSTOP]

    section('Registration quality')
    report('patients with a simulation and both radiographs', len(ai), 78, '{:d}')
    check('one postoperative comparison per patient', len(po) == len(ai),
          f'{len(po)} vs {len(ai)}')
    report('whole-image NCC, median', round(ai.whole_ncc.median(), 2), 0.85, '{:.2f}')
    report('whole-image NCC, lowest', round(ai.whole_ncc.min(), 2), 0.30, '{:.2f}')
    report('whole-image NCC, highest', round(ai.whole_ncc.max(), 2), 0.98, '{:.2f}')

    section('Region similarity: gradient-magnitude correlation (the reported metric)')
    g = {r: ai['gncc_' + r].median() for r, _label in REGIONS}
    for r, label in REGIONS:
        print(f"  {r:<18} {label:<46} median gncc = {g[r]:.3f}")
    # two decimal places are quoted in the manuscript, so a tolerance of 0.011
    # would let a genuine rounding disagreement through
    report('gncc, iliac wings and sacrum (median)', round(g['pelvis'], 2), 0.22,
           '{:.2f}', tol=0.005)
    report('gncc, cervicothoracic region (median)',
           round(g['cervical_shoulder'], 2), 0.34, '{:.2f}', tol=0.005)
    report('gncc, instrumented thoracolumbar segment (median)',
           round(g['spine_field'], 2), 0.07, '{:.2f}', tol=0.005)

    section('Is the difference between regions larger than chance? (AI simulations)')
    for r in ('pelvis', 'cervical_shoulder'):
        _z, p = wilcoxon_signed_rank(ai['gncc_' + r], ai.gncc_spine_field)
        check(f'gncc {r} > instrumented segment', g[r] > g['spine_field'],
              f'{g[r]:.3f} vs {g["spine_field"]:.3f}')
        report(f'  Wilcoxon p, {r} vs instrumented', '< 0.001' if p < 0.001 else round(p, 4),
               '< 0.001')
        print(f"      (exact p = {p:.3g})")

    section('Grey-value correlation and residual displacement show the same pattern')
    for r, _label in REGIONS:
        print(f"  {r:<18} ncc {ai['ncc_' + r].median():.3f}   "
              f"displacement {ai['disp_' + r + '_pctpelvis'].median():.2f}% of inter-iliac width")
    check('grey-value correlation is lowest in the instrumented segment',
          ai.ncc_spine_field.median() < ai.ncc_pelvis.median()
          and ai.ncc_spine_field.median() < ai.ncc_cervical_shoulder.median(),
          f'{ai.ncc_spine_field.median():.3f} vs '
          f'{ai.ncc_pelvis.median():.3f} / {ai.ncc_cervical_shoulder.median():.3f}')
    check('residual displacement is largest in the instrumented segment',
          ai.disp_spine_field_pctpelvis.median() > ai.disp_pelvis_pctpelvis.median(),
          f'{ai.disp_spine_field_pctpelvis.median():.2f}% vs '
          f'{ai.disp_pelvis_pctpelvis.median():.2f}%')
    _z, p = wilcoxon_signed_rank(ai.disp_pelvis_pctpelvis, ai.disp_spine_field_pctpelvis)
    report('  Wilcoxon p, displacement pelvis vs instrumented',
           '< 0.001' if p < 0.001 else round(p, 4), '< 0.001')

    section('The same analysis on the real postoperative radiograph')
    for r, _label in REGIONS:
        print(f"  {r:<18} simulation {ai['ncc_' + r].median():.3f}   "
              f"real postoperative {po['ncc_' + r].median():.3f}")
    lower = [r for r, _l in REGIONS if po['ncc_' + r].median() < ai['ncc_' + r].median()]
    check('grey-value correlation is lower for the real postoperative image in every region',
          len(lower) == len(REGIONS), f'lower in {len(lower)} of {len(REGIONS)} regions')

    section('What this does and does not establish')
    print("  These metrics are sensitive to the re-rendering of texture and to the lower\n"
          "  resolution of the generated images, and the regions were defined geometrically\n"
          "  rather than by anatomical landmarks. The comparison is exploratory and is not\n"
          "  used in the manuscript to support any claim of anatomical preservation.")

    finish('the manuscript')


if __name__ == '__main__':
    main()
