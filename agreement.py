"""Agreement and effect-size estimators used in this study.

Every estimator is written out from its definition rather than taken from a
library, so that the implementation is inspectable. The formulas are given in
the docstrings.

Reference for ICC(2,1): Shrout PE, Fleiss JL. Intraclass correlations: uses in
assessing rater reliability. Psychol Bull. 1979;86(2):420-428.
Reference for CCC: Lin LI-K. A concordance correlation coefficient to evaluate
reproducibility. Biometrics. 1989;45(1):255-268.
"""

import numpy as np
from scipy import stats


def icc21(x, y):
    """ICC(2,1): two-way random effects, absolute agreement, single measurement.

    With n subjects and k=2 measurements, and MSR / MSC / MSE the mean squares
    for subjects, measurements and residual:

        ICC(2,1) = (MSR - MSE) / (MSR + (k-1)*MSE + k*(MSC - MSE)/n)
    """
    m = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    n, k = m.shape
    grand = m.mean()
    ss_r = k * ((m.mean(1) - grand) ** 2).sum()
    ss_c = n * ((m.mean(0) - grand) ** 2).sum()
    ss_e = ((m - grand) ** 2).sum() - ss_r - ss_c
    msr, msc, mse = ss_r / (n - 1), ss_c / (k - 1), ss_e / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def icc21_ci(x, y, alpha=0.05):
    """Confidence interval for ICC(2,1) (Shrout & Fleiss 1979).

    With F = MSR/MSE having (n-1, (n-1)(k-1)) degrees of freedom,

        limit = (F / F_{p} - 1) / (F / F_{p} + k - 1)

    evaluated at p = alpha/2 for the upper limit and p = 1-alpha/2 for the
    lower one.
    """
    m = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    n, k = m.shape
    grand = m.mean()
    ss_r = k * ((m.mean(1) - grand) ** 2).sum()
    ss_c = n * ((m.mean(0) - grand) ** 2).sum()
    ss_e = ((m - grand) ** 2).sum() - ss_r - ss_c
    msr, msc, mse = ss_r / (n - 1), ss_c / (k - 1), ss_e / ((n - 1) * (k - 1))
    icc = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    df1, df2 = n - 1, (n - 1) * (k - 1)
    f = msr / mse
    lo, hi = stats.f.ppf(1 - alpha / 2, df1, df2), stats.f.ppf(alpha / 2, df1, df2)
    return icc, (f / lo - 1) / (f / lo + k - 1), (f / hi - 1) / (f / hi + k - 1)


def ccc_ci(x, y, alpha=0.05):
    """Lin's CCC with the confidence interval of Lin (1989), on the z scale."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    r = np.corrcoef(x, y)[0, 1]
    v = ccc(x, y)
    u = (mx - my) / np.sqrt(sx * sy)
    z = np.arctanh(v)
    k = ((1 - r ** 2) * v ** 2 / ((1 - v ** 2) * r ** 2)
         + 2 * u ** 2 * v ** 3 * (1 - v ** 2) / (r * (1 - r ** 2) ** 2)
         - u ** 4 * v ** 4 / (2 * r ** 2 * (1 - v ** 2) ** 2))
    se = np.sqrt(k / (n - 2))
    half = stats.norm.ppf(1 - alpha / 2) * se
    return v, np.tanh(z - half), np.tanh(z + half)


def ccc(x, y):
    """Lin's concordance correlation coefficient, using population variances.

        CCC = 2*cov(x,y) / [var(x) + var(y) + (mean(x) - mean(y))**2]
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=0), y.var(ddof=0)
    sxy = ((x - mx) * (y - my)).mean()
    return 2 * sxy / (vx + vy + (mx - my) ** 2)


def bland_altman(x, y, alpha=0.05):
    """Bias (mean difference y-x) with its CI, and the 95% limits of agreement."""
    d = np.asarray(y, float) - np.asarray(x, float)
    n = len(d)
    bias = d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    t = stats.t.ppf(1 - alpha / 2, n - 1)
    return {
        'n': n,
        'bias': bias,
        'bias_ci': (bias - t * se, bias + t * se),
        'loa': (bias - 1.96 * sd, bias + 1.96 * sd),
        'sd_diff': sd,
    }


def wilson(k, n, alpha=0.05):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (centre - half, centre + half)


def pearson_ci(x, y, alpha=0.05):
    """Pearson r with a Fisher-z confidence interval."""
    r, _ = stats.pearsonr(x, y)
    n = len(x)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - stats.norm.ppf(1 - alpha / 2) * se), np.tanh(z + stats.norm.ppf(1 - alpha / 2) * se)
    return r, (lo, hi)


def wilcoxon_signed_rank(x, y):
    """Wilcoxon signed-rank test for paired samples (normal approximation).

    Zero differences are discarded and the remaining |d| are ranked with ties
    averaged. With W the sum of the ranks of the positive differences,

        mu    = n(n + 1) / 4
        sigma = sqrt( n(n + 1)(2n + 1) / 24 - sum(t^3 - t) / 48 )

    the second sum running over the tie groups of size t. W is corrected
    towards mu by 0.5 (continuity), and the two-sided p comes from the normal
    tail. This is the same approximation scipy.stats.wilcoxon uses above n = 50.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    d = x - y
    d = d[d != 0]
    n = d.size
    if n == 0:
        return 0.0, 1.0
    r = stats.rankdata(np.abs(d))
    w = float(r[d > 0].sum())
    mu = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie = float((((counts ** 3) - counts) / 48.0).sum())
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie)
    if sigma == 0:
        return 0.0, 1.0
    z = (w - mu - 0.5 * np.sign(w - mu)) / sigma
    return float(z), float(2.0 * stats.norm.sf(abs(z)))


def cohens_kappa(a, b):
    """Cohen's kappa for two raters over a shared set of categories."""
    a, b = np.asarray(a), np.asarray(b)
    cats = np.unique(np.concatenate([a, b]))
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for u, v in zip(a, b):
        m[idx[u], idx[v]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(1) @ m.sum(0)) / n ** 2
    return (po - pe) / (1 - pe)


def paired_mean_ci(x, y, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap confidence interval for the paired mean difference.

    `x` and `y` are paired measurements on the same subjects; the quantity
    estimated is mean(x) - mean(y). Resampling is over subjects, so the pairing
    is preserved. The seed is fixed so that the interval is reproducible.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    d = x - y
    n = d.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), (float(lo), float(hi))


def mcnemar_exact(b, c):
    """Exact (binomial) McNemar test from the two discordant cell counts.

    `b` counts pairs where the first method met the threshold and the second did
    not; `c` counts the reverse. The two-sided p value is the exact binomial
    probability of a result at least as extreme as the smaller of the two counts
    under p = 0.5.
    """
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return float(min(1.0, 2 * tail))
