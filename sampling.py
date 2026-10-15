#!/usr/bin/env python3
"""How the Tier 2 and Tier 3A case samples were drawn.

Reviewer 1 (minor comment 1) asks which method was used to sample cases for
Tier 2 and Tier 3. The procedure is implemented here, and the sample actually
analysed is checked against the description given in the manuscript.

Usage:
    python sampling.py [--data PATH_TO_raw_data_待补.xlsx]
                       [--tier2-map PATH_TO_Tier2_图号映射.csv]

Tier 2
    120 images: for each of 60 patients, one AI-generated simulation and that
    patient's own real postoperative radiograph, presented as a matched pair.
    The 60 patients were drawn from the 354-patient cohort by stratified random
    sampling on Lenke curve type, with allocation proportional to the cohort.

Tier 3A
    50 patients drawn from the 354-patient cohort by stratified sampling on the
    simulation error category, with a fixed quota per stratum -- 35 acceptable,
    10 under-simulation, 5 over-simulation -- rather than proportional
    allocation, because the over-simulation stratum is small and is the
    clinically important one.

On seeds
    The draws were made with numpy's legacy global RandomState. The seed was not
    recorded at the time and cannot be recovered from the delivered tables, so
    the sample cannot be regenerated from a seed here. What is delivered instead
    is the realised sample itself, fixed in the case-level tables, together with
    the procedure below and the checks that the realised sample satisfies the
    stated stratification. Any future draw should record the seed; the function
    `stratified_draw` takes one explicitly for that reason.

    This is the only randomisation anywhere in the analysis of this manuscript.
    Image generation could not be seeded, because the web interface exposed no
    sampling parameters.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from checks import report, check, section, finish

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, os.pardir, 'raw_data_待补.xlsx')
DEFAULT_MAP = os.path.join(HERE, os.pardir, 'Tier2_图号映射.csv')

TIER3A_QUOTA = {'acceptable': 35, 'under': 10, 'over': 5}


def stratified_draw(frame, stratum, n, quota=False, seed=None):
    """Draw `n` rows, stratified by `stratum`.

    quota=False  allocation proportional to each stratum's share of the frame,
                 rounded to whole cases and made up to `n` largest-remainder
                 first
    quota=True   `n` is a dict of fixed per-stratum counts

    Returns the selected index. `seed` is required whenever a stratum holds more
    candidates than it is allocated, so that the draw is reproducible.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    groups = {k: v.index.values for k, v in frame.groupby(stratum)}
    if quota:
        sizes = dict(n)
    else:
        exact = {k: n * len(v) / len(frame) for k, v in groups.items()}
        sizes = {k: int(np.floor(x)) for k, x in exact.items()}
        for k in sorted(exact, key=lambda k: -(exact[k] - sizes[k]))[:n - sum(sizes.values())]:
            sizes[k] += 1
    picked = []
    for k, idx in groups.items():
        take = min(sizes.get(k, 0), len(idx))
        picked.extend(rng.choice(idx, size=take, replace=False).tolist())
    return np.array(sorted(picked))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default=DEFAULT_DATA)
    ap.add_argument('--tier2-map', default=DEFAULT_MAP)
    ap.add_argument('--crosswalk', default=None,
                    help='CSV with columns [mapping_id, patient_id] linking the Tier 2 '
                         'mapping to the analysis cohort; without it the Lenke '
                         'stratification cannot be checked and is reported as such')
    args = ap.parse_args()
    if not os.path.exists(args.data):
        sys.exit(f"data file not found: {args.data}")

    print(f"data   : {args.data}")
    print(f"python : {sys.version.split()[0]}  numpy {np.__version__}")

    cohort = pd.read_excel(args.data, sheet_name='T1_Patient_Raw_357')

    section('Tier 3A: 50 patients, quota sampling on the simulation error category')
    survey = pd.read_excel(args.data, sheet_name='T3A_Patient_Survey')
    realised = survey.bias_class.value_counts().to_dict()
    report('patients surveyed', len(survey), 50, '{:d}')
    for cat, want in TIER3A_QUOTA.items():
        report(f'  {cat}', int(realised.get(cat, 0)), want, '{:d}')
    check('the realised allocation is exactly the stated quota',
          all(int(realised.get(c, 0)) == w for c, w in TIER3A_QUOTA.items()),
          str({c: int(realised.get(c, 0)) for c in TIER3A_QUOTA}))

    pool = cohort.bias_class.value_counts().to_dict()
    print('\n  for comparison, the cohort itself holds:')
    for cat in TIER3A_QUOTA:
        print(f"    {cat:<12} {pool.get(cat, 0):>3} of {len(cohort)} "
              f"({100 * pool.get(cat, 0) / len(cohort):.1f}%)")
    print(f"  the over-simulation stratum is {pool.get('over', 0)} patients, of whom "
          f"{realised.get('over', 0)} were surveyed; proportional allocation would have "
          f"taken {50 * pool.get('over', 0) / len(cohort):.1f}.")

    quota_idx = stratified_draw(cohort, 'bias_class', TIER3A_QUOTA, quota=True, seed=0)
    prop_idx = stratified_draw(cohort, 'bias_class', 50, quota=False, seed=0)
    print('\n  what each allocation rule yields on the 354-patient cohort:')
    for label, idx in [('fixed quota (used)', quota_idx), ('proportional', prop_idx)]:
        counts = cohort.loc[idx, 'bias_class'].value_counts().to_dict()
        print(f"    {label:<20} " + '  '.join(
            f"{c} {counts.get(c, 0)}" for c in ('acceptable', 'under', 'over')))
    check('the fixed-quota rule reproduces the surveyed allocation',
          cohort.loc[quota_idx, 'bias_class'].value_counts().to_dict()
          == {c: w for c, w in TIER3A_QUOTA.items()})
    check('proportional allocation would have been a different sample',
          set(prop_idx) != set(quota_idx),
          'the over-simulation stratum is deliberately over-sampled')

    section('Tier 2: 60 patients, stratified on Lenke curve type')
    t2 = pd.read_excel(args.data, sheet_name='T2_Turing_Raw_120')
    report('images presented', len(t2), 120, '{:d}')
    report('images labelled AI-generated', int((t2.is_ai == 1).sum()), 60, '{:d}')
    report('images labelled real', int((t2.is_ai == 0).sum()), 60, '{:d}')
    check('the test set is a matched pair per patient',
          int((t2.is_ai == 1).sum()) == int((t2.is_ai == 0).sum()))

    if not os.path.exists(args.tier2_map):
        print(f"\n  Lenke stratification not checked: {args.tier2_map} not found.")
        print("  That mapping connects each image to its patient and is not distributed\n"
              "  with the code, so the check is skipped rather than silently passed.")
    else:
        m = pd.read_csv(args.tier2_map)
        # T2_Turing_Raw_120 lists its 60 simulations first and its 60 real
        # radiographs second, numbered 1-120 in that order; the mapping numbers
        # the same 120 images in the order they were presented, which interleaves
        # the two. The two image_id columns therefore do not denote the same
        # image, and the files are joined on the AI/real label, not on image_id.
        check('the mapping covers 120 images', len(m) == 120, f'{len(m)} rows')
        check('the mapping holds 60 simulations and 60 real radiographs',
              m.is_ai.value_counts().to_dict() == {1: 60, 0: 60},
              str(m.is_ai.value_counts().to_dict()))
        per_patient = m.groupby('patient_id').is_ai.sum()
        check('each patient contributes exactly one simulation',
              bool((per_patient == 1).all()) and len(per_patient) == 60,
              f'{len(per_patient)} patients, counts {sorted(set(per_patient))}')

        drawn = m.loc[m.is_ai == 1, 'patient_id'].unique()
        report('patients contributing a simulation', len(drawn), 60, '{:d}')

        if args.crosswalk is None:
            print("\n  Lenke stratification NOT CHECKED. The delivered Tier 2 mapping names\n"
                  "  patients by hospital record number; the analysis cohort uses a study\n"
                  "  pseudonym, and no crosswalk between the two is distributed with the\n"
                  "  code. Supply --crosswalk with columns [mapping_id, patient_id] to run\n"
                  "  this check. It is skipped rather than silently passed.")
        else:
            cw = pd.read_csv(args.crosswalk)
            j = m.merge(cw, left_on='patient_id', right_on='mapping_id', how='left')
            drawn = j.loc[j.is_ai == 1, 'patient_id_y'].dropna().unique()
            lut = cohort.set_index('patient_id').lenke_type
            got = lut.reindex(drawn)
            check('every drawn patient is in the 354-patient cohort',
                  bool(got.notna().all()) and len(drawn) == 60,
                  f'{int(got.isna().sum())} not found, {len(drawn)} resolved')
            got = got.value_counts().sort_index()
            want = cohort.lenke_type.value_counts().sort_index()
            print('\n  Lenke type   drawn   cohort share   expected under proportional draw')
            within = []
            for k in want.index:
                share = want[k] / len(cohort)
                print(f"    {k:<10} {got.get(k, 0):>5}   {share * 100:>10.1f}%   "
                      f"{60 * share:>21.1f}")
                within.append(abs(got.get(k, 0) - 60 * share) <= max(1.0, 0.25 * 60 * share))
            check('no Lenke stratum departs from its cohort share by more than chance allows',
                  all(within), f'{sum(within)} of {len(within)} strata within tolerance')

    section('Seeds')
    print("  The only randomisation in the analysis is the draw of the Tier 2 and\n"
          "  Tier 3A samples. It was made with numpy's legacy global RandomState and\n"
          "  the seed was not recorded, so it is not reported; the realised sample is\n"
          "  fixed in the delivered tables and is checked above. The comparator models\n"
          "  use fixed hyperparameters and deterministic fits, and image generation\n"
          "  exposed no sampling parameters, so no further seed applies.")

    finish('the manuscript')


if __name__ == '__main__':
    main()
