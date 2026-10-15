#!/usr/bin/env python3
"""Table 4: the comparator models.

Two correction-rate baselines and four information-matched models, all scored
against the same blinded reference measurement as the AI workflow, by
leave-one-out cross-validation over the 354 patients.

Usage:
    python comparators.py [--data PATH_TO_case_level_tables.xlsx]

What each comparator receives
    Correction-rate baselines
        The population mean correction rate, (pre - post) / pre, taken over the
        training patients only. The overall baseline uses one rate for everyone;
        the Lenke-specific baseline uses the mean rate of the training patients
        of that Lenke type, falling back to the overall rate when a fold leaves
        a group empty. Both receive less information than the model under
        evaluation, and are reported as a performance floor rather than as
        competitors.

    Information-matched models
        The same structured variables that were supplied in the prompt:
        preoperative major and minor Cobb angles, Lenke type, planned upper and
        lower instrumented vertebrae, number of planned fusion levels, age at
        surgery, sex, and Risser grade. Each model predicts the actual
        postoperative major curve angle.

        Multivariable linear regression (OLS)
        Ridge regression, regularisation strength chosen by generalized
        cross-validation over a 25-point grid from 1e-3 to 1e3
        A decision tree of maximum depth 2, minimum 10 samples per leaf
        Shallow gradient boosting: 100 depth-1 trees at learning rate 0.05

        Both tree models use fixed hyperparameters and no feature subsampling,
        so their fits are deterministic; `random_state` is set explicitly only
        as a matter of record. All preprocessing (standardisation, vertebral
        level coding) is fitted inside each leave-one-out training fold, so no
        information from the held-out patient enters model fitting.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from agreement import icc21, wilcoxon_signed_rank
from checks import report, check, section, finish

warnings.filterwarnings('ignore', category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, os.pardir, 'data', 'case_level_tables.xlsx')

RANDOM_STATE = 0          # inert for these two models; recorded for completeness
ALPHA_GRID = np.logspace(-3, 3, 25)


def correction_rate(pre, post):
    """(preoperative - postoperative) / preoperative, as a proportion."""
    return (pre - post) / pre


def correction_rate_baseline(pre, post, group=None):
    """Leave-one-out prediction from the mean correction rate of the other 353.

    With `group`, the rate is taken within the patient's own Lenke type, falling
    back to the overall rate when the held-out patient is the only member of it.
    """
    n = len(pre)
    cr = correction_rate(pre, post)
    pred = np.empty(n)
    for i in range(n):
        other = np.ones(n, dtype=bool)
        other[i] = False
        if group is not None:
            same = other & (group == group[i])
            other = same if same.any() else other
        pred[i] = pre[i] * (1.0 - cr[other].mean())
    return pred


def features(df):
    """The structured variables the prompt received, as a numeric design matrix."""
    return pd.DataFrame({
        'preoperative major curve (deg)': df.pre_cobb_main,
        'preoperative minor curve (deg)': df.pre_cobb_sec,
        'Lenke type': df.lenke_type,
        'upper instrumented vertebra': df.uiv.str[1:].astype(int),
        'lower instrumented vertebra': df.liv.str[1:].astype(int),
        'planned fusion levels': df.fusion_levels,
        'age at surgery (years)': df.age,
        'sex (male)': (df.sex == 'M').astype(int),
        'Risser grade': df.risser,
    })


def loo(model, X, y):
    """Leave-one-out predictions, with the model refitted on each 353-patient fold."""
    pipe = Pipeline([('standardise', StandardScaler()), ('model', model)])
    return cross_val_predict(pipe, X, y, cv=LeaveOneOut(), n_jobs=-1)


def score(name, actual, pred, want):
    """MAE, RMSE, threshold attainment and agreement against the manuscript row."""
    err = pred - actual
    got = {
        'MAE': round(np.abs(err).mean(), 2),
        'RMSE': round(np.sqrt((err ** 2).mean()), 2),
        'within +/-5 deg': round(100 * (np.abs(err) <= 5).mean(), 1),
        'within +/-10 deg': round(100 * (np.abs(err) <= 10).mean(), 1),
        'ICC(2,1)': round(icc21(actual, pred), 3),
    }
    print(f"\n  {name}")
    for key, want_v in want.items():
        report(f'{name}: {key}', got[key], want_v,
               '{:.3f}' if key == 'ICC(2,1)' else '{:.2f}'
               if key in ('MAE', 'RMSE') else '{:.1f}')
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default=DEFAULT_DATA)
    args = ap.parse_args()
    if not os.path.exists(args.data):
        sys.exit(f"data file not found: {args.data}")

    print(f"data   : {args.data}")
    print(f"python : {sys.version.split()[0]}  scikit-learn "
          f"{__import__('sklearn').__version__}  random_state = {RANDOM_STATE}")

    df = pd.read_excel(args.data, sheet_name='T1_Patient_Raw_357')
    actual = df.post_cobb_main.values
    pre = df.pre_cobb_main.values
    X, y = features(df), actual

    section(f'Table 4: comparator models (N = {len(df)}, leave-one-out)')

    print('\n  correction-rate baselines (LOO mean correction rate)')
    base = correction_rate_baseline(pre, actual)
    score('Overall correction-rate baseline', actual, base,
          {'MAE': 5.10, 'RMSE': 6.52, 'within +/-5 deg': 55.9,
           'within +/-10 deg': 86.2, 'ICC(2,1)': 0.238})
    lk = correction_rate_baseline(pre, actual, group=df.lenke_type.values)
    score('Lenke-specific correction baseline', actual, lk,
          {'MAE': 4.85, 'RMSE': 6.29, 'within +/-5 deg': 61.6,
           'within +/-10 deg': 89.3, 'ICC(2,1)': 0.390})

    print('\n  information-matched models (same structured variables as the prompt)')
    models = [
        ('Multivariable linear regression', LinearRegression(),
         {'MAE': 4.92, 'RMSE': 6.35, 'within +/-5 deg': 59.0,
          'within +/-10 deg': 89.3, 'ICC(2,1)': 0.341}),
        ('Ridge regression (generalized CV)',
         RidgeCV(alphas=ALPHA_GRID),
         {'MAE': 4.90, 'RMSE': 6.33, 'within +/-5 deg': 59.3,
          'within +/-10 deg': 89.0, 'ICC(2,1)': 0.309}),
        ('Decision tree (max_depth 2, min_samples_leaf 10)',
         DecisionTreeRegressor(max_depth=2, min_samples_leaf=10,
                               random_state=RANDOM_STATE),
         {'MAE': 5.23, 'RMSE': 6.76, 'within +/-5 deg': 56.8,
          'within +/-10 deg': 85.9, 'ICC(2,1)': 0.234}),
        ('Gradient boosting (100 depth-1 trees, lr 0.05)',
         GradientBoostingRegressor(n_estimators=100, max_depth=1,
                                   learning_rate=0.05, random_state=RANDOM_STATE),
         {'MAE': 4.94, 'RMSE': 6.34, 'within +/-5 deg': 59.9,
          'within +/-10 deg': 87.9, 'ICC(2,1)': 0.295}),
    ]
    preds = []
    for name, model, want in models:
        preds.append(score(name, actual, loo(model, X, y), want))

    section('Reference rows: the AI workflow')
    ai = df.ai_cobb_main.values
    score('AI best-of-three selected', actual, ai,
          {'MAE': 3.26, 'RMSE': 4.11, 'within +/-5 deg': 81.6,
           'within +/-10 deg': 97.5, 'ICC(2,1)': 0.850})

    # the manuscript states the AI advantage as an inequality over these models
    section('AI workflow vs each comparator (paired, N = %d)' % len(df))
    print('  the manuscript states the AI advantage as a range, not per model:')
    d_ai = np.abs(ai - actual)
    gains = []
    for (name, _model, _want), p in zip(models, preds):
        d_m = np.abs(p - actual)
        gain = d_m.mean() - d_ai.mean()
        gains.append(gain)
        _z, pv = wilcoxon_signed_rank(d_ai, d_m)
        check(f'{name}: AI lower MAE', d_ai.mean() < d_m.mean(),
              f'{d_ai.mean():.2f} vs {d_m.mean():.2f} deg')
        print(f"      MAE difference {gain:+.2f} deg; "
              f"Wilcoxon signed-rank p = {pv:.3g}")
    report('AI advantage, lower bound (deg)', round(min(gains), 2), 1.64, '{:.2f}')
    report('AI advantage, upper bound (deg)', round(max(gains), 2), 1.97, '{:.2f}')

    finish('the manuscript')


if __name__ == '__main__':
    main()
