# Analysis code

Code for the statistical analyses reported in

> *Communication-Oriented Postoperative Radiographic Simulation in Adolescent
> Idiopathic Scoliosis Using a General-Purpose Multimodal AI Model: A
> Retrospective Three-Tier Validation Study*

This package exists to answer Reviewer 1, minor comment 1 (reproducibility of
the statistical analysis), and to cover the comparator analyses added during
revision. It contains every estimator used in the manuscript, written out from
its definition rather than imported from a library, and it re-derives the
reported values from the case-level tables so that the numbers can be checked
rather than taken on trust.

**This repository contains code only.** No patient data, no images and no input
tables are distributed here. Every quantity the scripts recompute is read from
the case-level tables described under *Input tables* below, which are held by
the corresponding author.

## Contents

| File | Description |
| --- | --- |
| `checks.py` | Reporting helpers: print computed vs reported, collect mismatches, exit non-zero |
| `agreement.py` | ICC(2,1) and its confidence interval, Lin's concordance correlation coefficient and its interval, Bland–Altman analysis, Wilcoxon signed-rank test, Wilson score interval, Pearson *r* with a Fisher-z interval, Cohen's kappa |
| `reproduce.py` | Tiers 1–3, measurement reliability, the generation workflow, and the prompt-sensitivity analysis |
| `comparators.py` | Table 4: the two correction-rate baselines and the four information-matched models |
| `surgeon_comparison.py` | Table 4: the three independent surgeon estimates and the panel mean |
| `sampling.py` | How the Tier 2 and Tier 3A case samples were drawn, and checks on the realised samples |
| `preservation.py` | The exploratory regional registration comparison (images required upstream) |
| `run_all.sh` | Runs all five self-checks in sequence |
| `requirements.txt` | Pinned versions of the six dependencies |

## Running it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

./run_all.sh /path/to/raw_data_待补.xlsx --ablation /path/to/消融实验_原始记录.xlsx
```

or one analysis at a time:

```bash
python reproduce.py          --data /path/to/raw_data_待补.xlsx
python comparators.py        --data /path/to/raw_data_待补.xlsx
python surgeon_comparison.py --data /path/to/raw_data_待补.xlsx --surgeons /path/to/surgeon_estimation_case_level.csv
python sampling.py           --data /path/to/raw_data_待补.xlsx
python preservation.py       --metrics /path/to/preservation_metrics.csv
```

`--ablation` is optional; that section is skipped with a notice if the file is
not supplied. Run without `--data`, every script stops immediately with
`data file not found`. That is intended rather than a defect: the case-level
tables referred to above are not part of this repository.

Each script prints one line per reported quantity. Any value that differs from
the manuscript is marked `DIFF`, listed again in the summary, and makes the
script exit non-zero — so a silent mismatch is not possible. As delivered, all
five scripts run to completion with no `DIFF` lines.

## Input tables

`raw_data_待补.xlsx`, one sheet per analysis:

| Sheet | Rows | Used for |
| --- | --- | --- |
| `T1_Patient_Raw_357` | 354 | Tier 1 agreement; coronal-balance parameters; the comparator models |
| `T1b_逐次生成记录` | 1062 | generation attempts; failure screening; per-selector error |
| `T1b2_逐例汇总` | 354 | candidate-set composition; within-patient spread |
| `T1c_选择记录` | 354 | selection counts; agreement between the three selectors |
| `T1d_双测量者Cobb` | 354 | inter-reader measurement reliability |
| `T2_Turing_Raw_120` | 120 | Tier 2 forced-choice accuracy per rater |
| `T3A_Patient_Survey` | 50 | Tier 3A survey by simulation-error category |
| `T3B_Clinician_Raw` | 1062 | Tier 3B ratings; mixed-effects and GEE models |

These tables contain de-identified case-level measurements only — no images and
no patient identifiers. They are not included in this repository. They are
available from the corresponding author under the conditions set out in the
Data Availability statement; supplying them re-derives every value reported in
the manuscript.

Two further inputs are needed by single scripts and are likewise not
distributed: the surgeon-estimate table read by `surgeon_comparison.py` (one row
per patient, aligned with `T1_Patient_Raw_357`) and the per-image metrics table
read by `preservation.py`.

## Estimators

**ICC(2,1)** — two-way random effects, absolute agreement, single measurement.
With *n* subjects, *k* = 2 measurements, and MSR / MSC / MSE the mean squares
for subjects, measurements and residual:

```
ICC(2,1) = (MSR - MSE) / (MSR + (k-1)*MSE + k*(MSC - MSE)/n)
```

Its confidence interval uses `F = MSR/MSE` on (n-1, (n-1)(k-1)) degrees of
freedom, with `limit = (F/F_p - 1) / (F/F_p + k - 1)` at p = alpha/2 and
1 - alpha/2 (Shrout & Fleiss, *Psychol Bull* 1979;86:420–428).

**Lin's CCC** — `2·cov(x,y) / [var(x) + var(y) + (mean(x) - mean(y))²]`, with
population variances; the confidence interval follows Lin (1989) on the
z = arctanh(CCC) scale (*Biometrics* 1989;45:255–268).

**Bland–Altman** — bias is the mean of (simulated − actual) with a *t*-based
interval; the limits of agreement are bias ± 1.96 SD of the differences.

**Wilcoxon signed-rank** — zero differences are dropped and the remaining |d|
ranked with ties averaged; the statistic is compared with its normal
approximation, with a 0.5 continuity correction and the tie correction to the
variance. This is the test behind the paired error comparisons (selected vs
first output; selected vs each comparator).

**Proportions** — Wilson score intervals. **Forced-choice accuracy** is tested
against 0.5 with an exact binomial test. **Paired proportions** across prompt
conditions, and across the AI workflow against each comparator, use the exact
McNemar test on the discordant pairs. **Failure-category counts** across
templates use Fisher's exact test. **Survey domains** across simulation-error
categories use Kruskal–Wallis. **Tier 3B** uses a linear mixed model with a
random intercept for clinician, and a GEE logistic model clustered by clinician.

**Comparator models** — leave-one-out cross-validation, with standardisation
and vertebral-level coding fitted inside each training fold. The correction-rate
baselines predict `preoperative × (1 − mean correction rate of the other 353)`;
the Lenke-specific version takes that rate within the patient's own Lenke type.
The four information-matched models receive the same structured variables as the
prompt.

## Random seeds

The only randomisation anywhere in the analysis is the draw of the Tier 2 and
Tier 3A case samples. It was made with numpy's legacy global `RandomState`, and
the seed was not recorded at the time; it cannot be recovered from the delivered
tables, and it is therefore not reported here. What is delivered instead is the
sample itself, fixed in the case-level tables, together with the procedure
(`sampling.py`) and checks that the realised sample satisfies the stated
stratification. Any future draw should record its seed; `stratified_draw()`
takes one explicitly for that reason.

No other step is stochastic. The comparator models use fixed hyperparameters and
deterministic fits, and the image-generation step could not be seeded — the web
interface exposed no sampling parameters, which is stated in the manuscript.

## Coverage

| Manuscript quantity | Script |
| --- | --- |
| Tier 1 agreement, coronal balance, Bland–Altman, ICC, CCC, Pearson | `reproduce.py` |
| Inter-reader measurement reliability | `reproduce.py` |
| Generation counts, failure taxonomy, template comparison | `reproduce.py` |
| Human-in-the-loop workflow, selector agreement, geometric optimum | `reproduce.py` |
| Table 4 baselines and information-matched models | `comparators.py` |
| Table 4 surgeon estimates and panel mean | `surgeon_comparison.py` |
| Tier 2 per-rater accuracy, sensitivity, specificity | `reproduce.py` |
| Tier 3A survey tests | `reproduce.py` |
| Tier 3B mixed-effects and GEE | `reproduce.py` |
| Prompt-sensitivity threshold attainment | `reproduce.py` |
| Tier 2 and Tier 3A sampling | `sampling.py` |
| Regional registration metrics | `preservation.py` |

One step is not reproduced here. The registration itself — affinely registering
each simulation and each postoperative radiograph to the patient's own
preoperative radiograph by maximisation of mutual information — needs the image
files and a registration library, and is performed by a separate script that is
not distributed. `preservation.py` starts from the per-image metrics table that
registration produced and reproduces the reported regional medians and tests.
