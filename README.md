# LoanGuard: Auditing Credit-Risk Models for Leakage, Calibration, and Temporal Bias

[![CI](https://github.com/adityak1609/loanguard-credit-risk-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/adityak1609/loanguard-credit-risk-audit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**[Live Demo →](https://loan-d-qwshwdrewxtufmpqexbqd3.streamlit.app/)**

An audit-first credit-risk case study on 1.35M+ LendingClub loans—not just a
default predictor. It reproduces and fixes probability inflation, demonstrates
target leakage that raises AUC from 0.72 to 0.9995, and separates survivorship
bias from genuine temporal drift.

The interesting results here are not the AUC. They are what happens when you ask
three questions the headline metric hides: *how much does the model add over the
lender's existing scorecard, are the probabilities it emits real probabilities,
and does the out-of-time drop mean what it appears to mean.*

---

## Findings

### 1. A single feature gets most of the way there

`grade` and `int_rate` are LendingClub's own risk-model output, not borrower
attributes. Any model trained on them is partly re-learning an existing
scorecard, so the headline number is uninterpretable on its own.

| Feature set | Features | Test AUC | Δ vs grade-only |
|---|---|---|---|
| `grade_only` (logistic regression) | 1 | 0.6794 | — |
| `incumbent_only` (grade + sub_grade + int_rate) | 3 | 0.6968 | +0.0174 |
| **`applicant_only` (no incumbent signal)** | 43 | **0.7182** | **+0.0388** |
| `full` | 46 | 0.7243 | +0.0449 |
| `full` — scaled logistic regression | 46 | 0.7101 | +0.0307 |

Two things fall out. Borrower attributes alone (0.7182) beat LendingClub's entire
grading system (0.6968), so the model is not merely echoing the incumbent — it
substitutes for it. And adding grade on top of borrower attributes buys only
+0.006, meaning the incumbent scorecard carries almost no information the raw
application does not already contain.

The gradient-boosted model beats a *properly scaled* linear baseline by 0.014 AUC.
The original notebook reported a much larger gap, but its logistic regression was
fed unscaled features and never converged.

Reproduce: `python scripts/run_baselines.py` → `reports/baselines.csv`

### 2. The probabilities were inflated ~3x by class weighting

The earlier model set `scale_pos_weight` to the class ratio (4.01), which is a
ranking device: it deliberately biases predicted probabilities upward. The
Streamlit app then rendered that output as "probability of default".

| Model | AUC | Brier | ECE | Mean predicted | Observed |
|---|---|---|---|---|---|
| weighted (previous) | 0.7077 | 0.1557 | 0.0759 | 25.1% | 20.0% |
| unweighted | 0.7243 | 0.1427 | 0.0020 | 20.0% | 20.0% |
| unweighted + isotonic | 0.7242 | 0.1428 | 0.0019 | 20.0% | 20.0% |

The honest result is that isotonic regression was **not needed**. LightGBM
optimising log-loss is already well calibrated (ECE 0.0020); the entire
miscalibration was self-inflicted by the class weighting, which also cost 0.017
AUC. The calibrator is retained in the serving path because it is near-identity
here and provides a hook if the base rate shifts, but claiming it fixed the
problem would be wrong — removing `scale_pos_weight` did.

Reproduce: `python scripts/train_calibrate.py` → `reports/calibration.csv`

### 3. The cost-optimal threshold is flat, and dominated by an assumption

Loss given default, estimated empirically from charged-off loans, is **0.622**.
Against a mean forgone-interest cost of $4,274, that puts the false-negative to
false-positive ratio at only **2.1:1** — not the 10:1 that lending intuition
suggests, because these are high-interest unsecured loans where rejecting a good
borrower is genuinely expensive.

At the selected threshold (0.51, the midpoint of the near-optimal band) on the
test set: 96.4% approval rate, $1,893 expected loss per application, **$9.37M
saved** versus approving everyone across 201,803 applications.

But the curve is flat to within 1% across thresholds **0.41–0.61**. Reporting a
single argmin would be false precision — the validation argmin (0.44) and the
test argmin (0.48) differ by pure sampling noise. The threshold is reported as a
band for that reason.

What actually moves the decision is the LGD assumption, not the model:

| LGD | Threshold | Approval rate | Savings vs approve-all |
|---|---|---|---|
| 0.40 | 0.62 | 99.5% | $0.6M |
| 0.50 | 0.53 | 98.4% | $2.9M |
| **0.622 (empirical)** | **0.48** | **96.4%** | **$9.4M** |
| 0.70 | 0.41 | 92.0% | $16.8M |
| 0.80 | 0.36 | 87.4% | $34.3M |

Any effort spent tightening the model is worth less than getting LGD right.

Reproduce: `reports/decision.json`, `reports/lgd_sensitivity.csv`

### 4. The out-of-time drop is a sampling artefact, not distribution shift

Training on ≤2014 and testing on 2016–2018 gives AUC 0.7359 → 0.7121, a drop of
**0.0238**. The tempting conclusion is model decay.

It isn't. The snapshot ends 2018Q4, so **no loan issued from 2016 onward has
reached the end of its term** — `matured_share` is exactly 0.0000 for 2016, 2017
and 2018. Filtering to resolved statuses therefore selects only loans that
terminated *early*, and early termination is heavily skewed toward default.

2014 and 2015 contain both matured and un-matured resolved loans, so the same
model scored on each subset isolates the effect:

| Year | Matured | n | Default rate | AUC |
|---|---|---|---|---|
| 2014 | no | 60,533 | 31.1% | 0.6950 |
| 2014 | yes | 162,570 | 13.7% | 0.7230 |
| 2015 | no | 92,520 | 36.4% | 0.6751 |
| 2015 | yes | 283,026 | 14.9% | 0.6970 |

Un-matured loans default at 2.3x the rate of matured ones, and the model scores
**0.0250 AUC worse** on them — within the same origination years, where there is
no distribution shift at all.

That 0.0250 maturity penalty accounts for essentially the entire 0.0238
out-of-time drop. **There is no evidence of genuine temporal degradation in this
data; the apparent decay is survivorship.**

Reproduce: `python scripts/temporal_validation.py` → `reports/maturity_confound.csv`

### 5. With the bias removed, half the drop survives

Finding 4 argued from a proxy. The direct fix is a **fixed observation window**:
relabel the target as *defaulted within 12 months of origination*, dating the
event from `last_pymnt_d` plus a 3-month delinquency lag. Every vintage is then
watched for exactly the same 12 months, loans still performing at the snapshot
become valid negatives instead of being discarded, and eligibility depends only
on issue date — never on outcome.

This recovers **1,765,426 loans**, up from 1,345,350, and the vintage curve
changes completely:

| Issue year | Default rate, resolved-only | Default rate, 12-month window |
|---|---|---|
| 2013 | 15.6% | 3.2% |
| 2014 | 18.5% | 3.4% |
| 2015 | 20.2% | 4.0% |
| 2016 | 23.3% | 4.7% |
| 2017 | 23.1% | 4.4% |

The resolved-only view suggests credit quality collapsing after 2015. It wasn't
— that curve is measuring how much of each vintage had resolved, not how much
had defaulted.

Re-running the out-of-time test on this target (train ≤2014, validate 2015, test
2016–17):

| Target | In-time AUC | Out-of-time AUC | Drop |
|---|---|---|---|
| resolved-only | 0.7359 | 0.7121 | **0.0238** |
| 12-month window | 0.7249 | 0.7119 | **0.0131** |

**The honest result is that this partly contradicts finding 4.** Roughly half
the apparent degradation was survivorship, but a residual 0.0131 survives the
correction — so there is some mild genuine drift, not none. The per-vintage AUC
trend (0.79 in 2014 → 0.72 in 2015 → 0.71 in 2016–17) points the same way.
Finding 4's maturity decomposition was directionally right about the mechanism
and overstated how much it explained.

The delinquency lag is the softest assumption; it moves the base rate but not
the conclusion:

| Lag (months) | Default rate |
|---|---|
| 0 | 6.13% |
| 3 (used) | 4.13% |
| 5 | 2.88% |

Reproduce: `python scripts/horizon_validation.py` → `reports/horizon_validation.json`

### 6. Published 0.95+ AUCs on this dataset are leakage

The dataset ships with columns recorded *after* the loan performed:
`last_fico_range_low/high` (credit score at the most recent pull — for a
defaulted loan, pulled after the default), `last_pymnt_amnt`, `out_prncp`,
`total_pymnt`, `total_rec_int`. None exist at decision time.

Adding them to the same model on the same split:

| Feature set | Test AUC | Average precision | Deployable |
|---|---|---|---|
| `full` | 0.7243 | 0.3943 | yes |
| `full_leaky` | **0.9995** | 0.9987 | **no** |

`total_pymnt` is the top feature by a wide margin, which is unsurprising — a
fully repaid loan pays back principal plus interest and a charged-off one does
not, so the column is very nearly the label itself.

This is in the repo as a runnable script rather than a caveat, because "our 0.72
is honest and their 0.95 is leakage" is a claim worth being able to demonstrate.

Reproduce: `python scripts/leakage_demo.py` → `reports/leakage_demo.csv`

### 7. Geography is excluded from the model and audited separately

`addr_state` and masked `zip_code` are retained only for post-model monitoring.
They are excluded from every deployable `FeatureSpec` because geography can act
as a proxy for protected characteristics. The audit reports approval rate,
observed default rate, good-applicant approval rate, bad-loan approval rate, and
the four-fifths screening ratio for sufficiently large groups.

| Proxy | Groups with ≥500 test loans | Minimum approval-rate ratio | Below 0.80 |
|---|---:|---:|---:|
| State | 43 | 0.961 | 0 |
| Masked ZIP | 110 | 0.956 | 0 |

At the 0.51 decision threshold, the held-out approval rate is 96.39%. No
eligible geographic group triggers the four-fifths screen, although the high
overall approval rate makes this a weak stress test of disparate rejection.

This is deliberately framed as a **proxy disparity screen**, not a compliance
claim: the extract has no direct protected-class labels, so state and masked ZIP
cannot establish demographic fairness. See `MODEL_CARD.md` for the use boundary
and the controls a real underwriting deployment would still require.

Reproduce: `python scripts/fairness_audit.py` →
`reports/fairness_{addr_state,zip_code}.csv` and `reports/fairness_summary.json`

---

## What changed from the original pipeline

| | Before | After |
|---|---|---|
| Rows | 1,220,127 | 1,345,350 resolved-only, or 1,765,426 on the 12-month window target (`dropna` removed — LightGBM handles NaN, and `emp_length` missingness correlates with the target) |
| Target | terminal status, unobservable for late vintages | also "default within 12 months of origination", comparable across vintages |
| Leakage | untested | demonstrated explicitly (0.7243 → 0.9995) |
| Baseline | none | grade-only, incumbent-only, applicant-only ablations |
| Linear baseline | unscaled, non-converged | imputed + scaled pipeline |
| Threshold | 0.5, implicit in `predict()` | expected-loss minimisation with per-loan costs |
| Headline metric | F1 = 0.427 at p=0.5 | cost per application, approval rate, calibration |
| Calibration | unmeasured, ~3x inflated | ECE 0.0020, reported before/after |
| Validation | random split only | random + out-of-time + maturity decomposition |
| Serving | column list retyped by hand in the app | shared `FeatureSpec` artefact |
| Features | encoding only | leverage, payment burden, credit history, log income |
| Fairness | not evaluated | geographic proxies excluded from features and monitored by group |
| Engineering | no automated checks | unit tests + lint + GitHub Actions CI |

Two live bugs in the Streamlit app were fixed along the way: it displayed
uncalibrated scores as probabilities, and its SHAP waterfall used
`expected_value[0]`, which explains the **repayment** class — every contribution
was shown with the wrong sign.

---

## Layout

```
src/loanguard/     config, data prep, features, model, evaluation, fairness
scripts/           prepare_data · run_baselines · train_calibrate
                   temporal_validation · horizon_validation · leakage_demo
                   fairness_audit
tests/             unit tests for data, feature parity, costs, and fairness
reports/           all generated results (tracked in git)
notebooks/legacy/  original exploratory notebooks, superseded
streamlit_app.py   serving UI (deployed on Streamlit Community Cloud)
MODEL_CARD.md      intended use, performance, subgroup audit, limitations
TECHNICAL.md       how the code produces these findings, and where it is soft
TECH_STACK.md      what each library does and what the model looks at
processed/         model artefacts tracked for deployment (parquet files gitignored)
```

`TECHNICAL.md` is the deep dive: the two target definitions, the cost model
derivation, the train/serve contract, and a ranked list of known soft spots.
`TECH_STACK.md` is the plain-English version — no prior background assumed.

## Running it

```bash
pip install -r requirements.txt

# Download accepted_2007_to_2018Q4.csv (~1.6 GB) into the repo root:
# https://www.kaggle.com/datasets/wordsforthewise/lending-club

python scripts/prepare_data.py        # ~4 min, writes both target tables
python scripts/run_baselines.py       # finding 1
python scripts/train_calibrate.py     # findings 2-3, writes serving artefacts
python scripts/temporal_validation.py # finding 4
python scripts/horizon_validation.py  # finding 5 (needs finding 4's output)
python scripts/leakage_demo.py        # finding 6
python scripts/fairness_audit.py      # finding 7

pip install -r requirements-dev.txt
ruff check src scripts tests streamlit_app.py
pytest

streamlit run streamlit_app.py
```

Seed is fixed at 42 throughout (`src/loanguard/config.py`).

## Deployment

The app is deployed on **Streamlit Community Cloud** at:

> https://loan-d-qwshwdrewxtufmpqexbqd3.streamlit.app/

The four serving artefacts (`lgbm_model.pkl`, `calibrator.pkl`, `feature_spec.json`,
`serving_config.json`) are tracked in `processed/` so that Streamlit Cloud can load
them directly from the repo. The large parquet data files remain gitignored.

To redeploy after changes, push to `main` — Streamlit Cloud auto-redeploys on every
commit.

## Known limitations

- **The residual 0.013 out-of-time drop is unexplained.** Finding 5 removes
  survivorship but leaves real degradation on the table. Candidate causes:
  LendingClub's borrower mix widening after 2015, or the newer bureau fields
  (`bc_util`, `tot_cur_bal`, `acc_open_past_24mths`) being absent before 2012,
  which makes early vintages structurally different. Not yet separated.
- **The delinquency lag is an assumption, not a measurement.** 3 months is a
  reasonable proxy for last-payment → 90+ DPD, but the dataset does not date
  the delinquency directly.
- **The 12-month horizon excludes 2018 originations** and truncates late
  defaults. A 24-month window would capture more of the loss but cover fewer
  vintages.
- **LGD is a portfolio average.** It almost certainly varies by grade and term;
  a per-segment estimate would sharpen the cost model more than any modelling
  change (see the sensitivity table).
- **`cost_fp` assumes a declined applicant is a total loss of interest.** In
  practice they may be re-priced rather than declined outright.
- **The fairness audit is proxy-only.** State and masked ZIP can reveal
  geographic disparities but are not protected-class labels. The audit cannot
  establish demographic parity or legal compliance, and feature exclusion does
  not remove proxy information carried by correlated financial variables.
