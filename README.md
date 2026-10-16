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
| `figstats.py` | Recomputes every statistic the figures plot and writes `figstats.json` |
| `figstyle.py` | Shared plotting style, palette and input path for the figure scripts |
| `make_fig3.py` | Figure 3 — structural agreement (scatter, Bland–Altman, error distribution) |
| `make_fig5.py` | Figure 5 — perceived potential communication utility |
| `make_supp.py` | Supplementary Figures S1–S5 |
| `requirements.txt` | Pinned versions of the six dependencies |

## Running it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

./run_all.sh /path/to/case_level_tables.xlsx --ablation /path/to/ablation_record.xlsx
```

or one analysis at a time:

```bash
python reproduce.py          --data /path/to/case_level_tables.xlsx
python comparators.py        --data /path/to/case_level_tables.xlsx
python surgeon_comparison.py --data /path/to/case_level_tables.xlsx --surgeons /path/to/surgeon_estimation_case_level.csv
python sampling.py           --data /path/to/case_level_tables.xlsx
python preservation.py       --metrics /path/to/registration_metrics.csv
```

To regenerate the figures:

```bash
python figstats.py --data /path/to/case_level_tables.xlsx --out figstats.json
python make_fig3.py --outdir .
python make_fig5.py --outdir .
python make_supp.py --outdir .
```

`figstats.py` writes `figstats.json`, which the figure scripts read; a copy computed
from the delivered tables is included, so the three `make_*` scripts run on their own.

`--ablation` is optional; that section is skipped with a notice if the file is
not supplied. Run without `--data`, every script stops immediately with
`data file not found`. That is intended rather than a defect: the case-level
tables referred to above are not part of this repository.

Every script has a default input path of `../data/<name>` relative to this
directory, where `<name>` is the generic filename listed under *Input tables*
below. Putting the tables in a `data/` folder alongside this package therefore
lets the scripts run with no arguments; otherwise pass the paths explicitly as
above. The filenames are generic placeholders — the tables may be stored under
any names, and no directory layout is assumed beyond what you pass in.

Each script prints one line per reported quantity. Any value that differs from
the manuscript is marked `DIFF`, listed again in the summary, and makes the
script exit non-zero — so a silent mismatch is not possible. As delivered, all
five scripts run to completion with no `DIFF` lines.

Lines beginning `--` are quantities the scripts compute for information and that
the manuscript does not report. They are printed, never compared, and are marked
`not reported` so that they cannot be mistaken for published values. Two are
printed by `reproduce.py`: the geometric reference when candidates removed by the
failure screen are included, which a safe workflow could never reach and which is
therefore an upper bound rather than a result; and the error a different selector
would have obtained had they chosen instead of the primary selector, which
quantifies how much the selection step depends on who performs it. Neither appears
in the manuscript, the Supplementary Materials or the response to reviewers.

## Input tables

`case_level_tables.xlsx`, one sheet per analysis:

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
distributed: `surgeon_estimation_case_level.csv`, read by
`surgeon_comparison.py` (one row per patient, aligned with `T1_Patient_Raw_357`),
and `registration_metrics.csv`, the per-image metrics table read by
`preservation.py`. `ablation_record.xlsx` holds the prompt-template ablation
records, and `tier2_image_mapping.csv` maps the Tier 2 images to their cases.

The sheet and column names inside these tables are quoted verbatim in the
scripts, since they are how the tables are read; they are the only place where
the delivery format of the source tables shows through.

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

**Two defined conventions.** The *first generated output* is the first of the
three attempts for that patient, whatever the failure screen later decided about
it — the zero-curation comparator. An attempt that yielded no measurable
residual curve counts as a miss for the threshold proportion and is excluded
from the mean absolute error, so the proportion keeps the full cohort as its
denominator. The *geometric reference* is the candidate, among those that
**passed** the failure screen, whose simulated Cobb angle was closest to the
true postoperative value. Screen-failed candidates are excluded because a
workflow that screens failures can never select them; including them would
report a ceiling the workflow cannot reach. The screen-inclusive figure is
printed alongside it, so the difference is visible rather than implied.

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
| Generation counts, failure taxonomy, measurability of failures | `reproduce.py` |
| Human-in-the-loop workflow, selector agreement, geometric optimum | `reproduce.py` |
| Table 4 baselines and information-matched models | `comparators.py` |
| Table 4 surgeon estimates and panel mean | `surgeon_comparison.py` |
| Tier 2 per-rater accuracy, sensitivity, specificity | `reproduce.py` |
| Tier 3A survey tests | `reproduce.py` |
| Tier 3B mixed-effects and GEE | `reproduce.py` |
| Prompt-sensitivity threshold attainment | `reproduce.py` |
| Tier 2 and Tier 3A sampling | `sampling.py` |
| Regional registration metrics | `preservation.py` |
| Figures 3 and 5 | `figstats.py`, `make_fig3.py`, `make_fig5.py` |
| Supplementary Figures S1–S5 | `figstats.py`, `make_supp.py` |

Figure 1 carries no convention-dependent number, and its flow-chart counts are
stated in the manuscript; the two in-figure text labels that were corrected during
revision (the Tier 1 heading in Figure 1 and one p value in Figure 4) were edited
directly in the image files, which are author-supplied and not part of this package.

One step is not reproduced here. The registration itself — affinely registering
each simulation and each postoperative radiograph to the patient's own
preoperative radiograph by maximisation of mutual information — needs the image
files and a registration library, and is performed by a separate script that is
not distributed. `preservation.py` starts from the per-image metrics table that
registration produced and reproduces the reported regional medians and tests.
