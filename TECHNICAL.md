# LoanGuard — Technical Documentation

Internal deep-dive. The `README.md` states *what was found*; this document explains
*how the code produces those findings*, why each design choice was made, and where
the remaining soft spots are.

Read order if you're new: §1 (mental model) → §3 (targets) → §7 (cost model) → §8
(the six experiments) → §10 (known soft spots).

---

## 1. The mental model

Most credit-risk projects are structured as *"fit a model, report AUC."* This one is
structured as **a chain of questions about whether a reported number means anything**.
The model itself is deliberately unremarkable — a stock LightGBM with hand-set
hyperparameters, never tuned. Everything interesting lives in the evaluation layer.

The six questions, in the order the code answers them:

| # | Question | Answered by |
|---|---|---|
| 1 | Does the model beat the lender's *existing* scorecard? | `run_baselines.py` |
| 2 | Are the outputs actually probabilities? | `train_calibrate.py` |
| 3 | What threshold does the *economics* imply, not `predict()`? | `train_calibrate.py` |
| 4 | Does the out-of-time drop mean decay, or is the label biased? | `temporal_validation.py` → `horizon_validation.py` |
| 5 | Why do published results on this data hit 0.95? | `leakage_demo.py` |
| 6 | Do approval outcomes vary across available geographic proxies? | `fairness_audit.py` |

A seventh implicit question — *is the serving path faithful to the training path?* — is
answered structurally by the `FeatureSpec` artifact (§9).

---

## 2. Repository layout and data flow

```
accepted_2007_to_2018Q4.csv      raw Kaggle extract, ~1.6 GB, 151 columns
        │
        │  scripts/prepare_data.py
        ▼
processed/loans.parquet          1,345,350 rows · resolved-only target
processed/loans_12m.parquet      1,765,426 rows · 12-month-window target
        │
        ├── run_baselines.py        → reports/baselines.{csv,json}
        ├── train_calibrate.py      → reports/calibration.csv, decision.json,
        │                             cost_curve_test.csv, lgd_sensitivity.csv,
        │                             reliability_*.csv
        │                           → processed/lgbm_model.pkl, calibrator.pkl,
        │                             feature_spec.json, serving_config.json
        ├── temporal_validation.py  → reports/temporal.json, temporal_per_year.csv,
        │                             maturity_confound.csv
        ├── horizon_validation.py   → reports/horizon_validation.json,
        │                             horizon_per_year.csv   [needs temporal.json]
        └── leakage_demo.py         → reports/leakage_demo.{csv,json}
                                       │
                                       ▼
                              streamlit_app.py  (loads the four serving artifacts)
```

`src/loanguard/` is the library; `scripts/` are thin drivers that own no logic beyond
orchestration and reporting. Every script re-derives its split from the same seeded
function, so results are comparable across scripts without passing state between them.

**Only one cross-script dependency exists:** `horizon_validation.py` reads
`reports/temporal.json` to print the resolved-only drop alongside its own. Run
`temporal_validation.py` first.

### Module responsibilities

| Module | Owns |
|---|---|
| `config.py` | Paths, seed, column groups, snapshot date, economic constants. No logic. |
| `data.py` | Raw load, type coercion, **both target definitions**, LGD estimation, per-loan cost attachment. |
| `features.py` | Feature engineering, the five named feature sets, and `FeatureSpec` (the train/serve contract). |
| `model.py` | Hyperparameters, the seeded 70/15/15 split, LightGBM and logistic-regression fitters. |
| `evaluate.py` | Ranking metrics, calibration metrics, the cost curve, threshold selection, LGD sensitivity. |
| `fairness.py` | Group approval/error metrics and the four-fifths proxy screen. |

---

## 3. The two targets (the heart of the project)

This is the single most important section. The project defines the label **twice**,
and the difference between them *is* findings 4 and 5.

### 3.1 Target A — terminal status (`data.build`)

```python
df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])]
df["target"] = df["loan_status"].isin(["Charged Off", "Default"]).astype(int)
```

Standard practice, and **structurally biased**. The reasoning:

- The snapshot ends **2018-12-31** (`config.SNAPSHOT_END`).
- A 36-month loan issued in 2017 matures in 2020 — after the snapshot.
- So it can only appear in the filtered data if it *ended early*.
- Early endings on unsecured consumer loans are dominated by charge-offs.
- ⇒ Filtering on terminal status **selects on the outcome** for late vintages.

`build()` makes the bias measurable rather than hidden by computing a `matured` flag:

```python
df["maturity_date"] = df["issue_d"] + pd.to_timedelta(df["term_months"] * 30.44, unit="D")
df["matured"] = df["maturity_date"] <= snapshot
```

(`30.44` is the mean days-per-month; `term` is only ever 36 or 60 months, so the
approximation is worth at most a couple of days and never flips the comparison.)

`matured_share` is **exactly 0.0000** for 2016, 2017 and 2018 in
`reports/data_summary.json` — the smoking gun for finding 4.

### 3.2 Target B — fixed observation window (`data.build_horizon`)

> *"Did this loan default within 12 months of origination?"*

Three properties make this the honest target:

**(a) Eligibility never touches the outcome.**
```python
window_end = df["issue_d"] + pd.DateOffset(months=horizon)
df = df[window_end <= snapshot]
```
A loan is in-sample iff it was issued early enough to be *watchable* for 12 months.
That is a function of `issue_d` and the snapshot date only. Contrast with target A,
where inclusion depends on `loan_status`.

**(b) Still-performing loans become valid negatives.**
Target A discards a 2016 loan that is current at the snapshot. Target B labels it 0 —
correctly, since it demonstrably did not default in its first 12 months. This is where
the extra **420,076 rows** come from (1,765,426 vs 1,345,350).

**(c) Every vintage is observed for an identical duration**, so vintage-over-vintage
default rates are actually comparable.

**Dating the default event.** The dataset gives no delinquency date, so it's inferred:

```python
stopped_paying = df["last_pymnt_d"].fillna(df["issue_d"])   # never paid ⇒ default from day 0
event_date     = stopped_paying + pd.DateOffset(months=dpd_lag)   # dpd_lag = 3
df["target"] = (is_default_status & (event_date <= issue_d + 12 months)).astype(int)
```

Two things to be clear about:

1. **`last_pymnt_d` is a post-origination column — is this leakage?** No. It is used to
   construct the *label*, never as a feature. Labels are always drawn from the future;
   that is what makes them labels. The test is whether the column reaches the model's
   input, and it does not: `last_pymnt_d` lives in `config.META` and appears in no
   entry of `FEATURE_SETS`. The genuinely leaky columns are quarantined separately in
   `config.LEAKY` (§8.5).
2. **The 3-month lag is an assumption, not a measurement.** `DEFAULT_EVENT_STATUSES`
   counts "Late (31-120 days)" as a default but deliberately excludes "In Grace Period"
   and "Late (16-30 days)", both of which are under 31 DPD and frequently cure.
   `data.horizon_sensitivity()` quantifies the assumption's cost by rebuilding the
   whole label at lags 0/3/5 → base rate 6.13% / 4.13% / 2.88%
   (`reports/horizon_sensitivity.csv`). The base rate moves; the drift conclusion does not.

### 3.3 Which target does what

| | Target A (resolved) | Target B (12-month) |
|---|---|---|
| Rows | 1,345,350 | 1,765,426 |
| Base rate | 19.96% | 4.13% |
| Used by | baselines, calibration, cost model, **the deployed model** | horizon validation only |
| Bias | survivorship on 2016+ | none known |

⚠️ **The shipped model is trained on target A.** See §10.1 — this is the most important
open issue in the project.

---

## 4. Data loading and coercion

`load_raw()` reads only `config.USECOLS` (~30 of 151 columns) — the raw file is 1.6 GB
and a full read is the difference between 4 minutes and an OOM.

`_coerce_numeric()` is shared by both targets so they can never drift apart:

| Column | Treatment | Why |
|---|---|---|
| `int_rate`, `revol_util` | strip `%` → float | stored as `"13.56%"` strings |
| `earliest_cr_line` | `%b-%Y` datetime | e.g. `"Aug-2001"` |
| `emp_length` | `EMP_LENGTH_MAP` → 0–10 | `"< 1 year"` → 0, `"10+ years"` → 10 |
| `grade` | `A–G` → 1–7 | genuinely ordinal; one-hot would discard the ordering |
| `sub_grade` | `grade*5 + digit` → 1–35 | `"C3"` → 3·5+3 = 18; monotone in risk |
| `term_months` | regex `(\d+)` | stored as `" 36 months"` |

### The `dropna` decision

The original notebooks called `df.dropna()`, losing ~125K rows. That was removed, for
two reasons:

1. LightGBM handles NaN natively — it learns a default split direction for missing
   values rather than requiring imputation.
2. **Missingness here is informative.** `emp_length` is absent for 5.8% of rows and is
   absent disproportionately for applicants without steady employment — precisely the
   riskier slice. `dropna()` silently deletes part of the signal *and* biases the base
   rate downward.

Missingness shares are recorded in `reports/data_summary.json` rather than being
imputed away. The logistic-regression baseline *does* impute (median) because it must
— see §6.2.

---

## 5. Feature engineering

`features.engineer()` adds six derived columns. All are ex-ante — computable from an
application form plus a credit pull.

```python
loan_to_income                = loan_amnt / annual_inc
installment_to_monthly_income = installment / (annual_inc / 12)     # ≈ payment burden
revol_bal_to_income           = revol_bal / annual_inc
log_annual_inc                = log1p(clip(annual_inc, 0, None))
credit_history_years          = (issue_d - earliest_cr_line).days / 365.25
term_months                   = parsed from `term` in _coerce_numeric
```

Notes:

- `annual_inc.replace(0, np.nan)` guards the three ratios against divide-by-zero;
  the result is NaN, which LightGBM handles.
- `log_annual_inc` exists **for the linear baseline's benefit**, not the tree's. GBMs
  are invariant to monotone transforms; logistic regression is not, and self-reported
  income on this dataset has absurd extremes (multi-million-dollar entries).
  Including it makes the linear baseline a fair opponent instead of a strawman.
- `installment_to_monthly_income` is the closest thing here to a real underwriting
  ratio (front-end DTI), and it is the engineered feature that matters most.

### The five feature sets

Defined once in `features.FEATURE_SETS`, consumed by every script:

| Set | Contents | Purpose |
|---|---|---|
| `grade_only` | `grade` (1 col) | Floor. What the lender already knows. |
| `incumbent_only` | `grade`, `sub_grade`, `int_rate` (3) | The *entire* existing scorecard. |
| `applicant_only` | borrower + engineered (43) | Can we underwrite without the incumbent? |
| `full` | everything ex-ante (46) | The production model. |
| `full_leaky` | `full` + `config.LEAKY` (52) | **Not a model.** Demo only. |

The split between `INCUMBENT` and `APPLICANT_*` in `config.py` is the conceptual core
of finding 1: `grade` and `int_rate` are *LendingClub's model output*, not borrower
attributes. Training on them and reporting AUC measures how well you re-learned
someone else's scorecard.

---

## 6. Model and split

### 6.1 The split

```python
train, temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df["target"])
val,   test = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp["target"])
```

70/15/15, stratified, seed 42, defined once in `model.split()`. Because it is a pure
function of the dataframe and a fixed seed, every script that calls it gets the *same*
partition without any script needing to persist indices.

**This is a random split, not temporal.** That is intentional: `run_baselines`,
`train_calibrate` and `leakage_demo` are answering questions about feature value,
calibration and economics, where a random split is the right control. The two temporal
scripts override it with explicit year-based slices.

### 6.2 Hyperparameters

```python
n_estimators=500, learning_rate=0.03, max_depth=8, num_leaves=50,
min_child_samples=20, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
random_state=42
```

Fixed, never tuned, with `early_stopping(50)` on validation AUC. This is a deliberate
stance: the project's claim is about *evaluation methodology*, and the LGD sensitivity
table (§7.4) shows the economics move far more with the LGD assumption than any
plausible hyperparameter gain would move them. Tuning would have added noise to the
comparisons without changing a single conclusion.

### 6.3 The logistic baseline

```python
make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
              LogisticRegression(max_iter=2000))
```

The original notebook fed **raw, unscaled** features to `lbfgs`, which did not converge
— making the linear baseline look far worse than it is and inflating the apparent value
of the tree model. Fixed, the honest gap is **0.7243 vs 0.7101 = +0.0142 AUC**, not the
large margin originally reported. Reporting a real gap of 0.014 instead of a fake gap
of 0.10 is the point.

---

## 7. The cost model

The original pipeline reported F1 at p = 0.5. Both halves of that are wrong: F1 assumes
false positives and false negatives cost the same, and 0.5 is just sklearn's
`predict()` default, not a decision.

### 7.1 Per-loan costs (`data._attach_economics`, `data.add_costs`)

```python
forgone_interest = clip(installment * term_months - loan_amnt, lower=0)

cost_fn = loan_amnt * LGD      # approved a loan that defaulted → principal lost
cost_fp = forgone_interest     # declined a loan that would have repaid → interest lost
```

Both are computed **only from ex-ante quantities** — amount, installment, term. No
realized-payment column is used. This matters: using `total_pymnt` to compute the loss
would smuggle leakage into the cost model even with a clean feature set.

Costs are per-loan, not constants. A declined \$35K applicant and a declined \$1K
applicant do not cost the same, and the threshold search accounts for that.

### 7.2 LGD (`data.estimate_lgd`)

```python
d = df[df["target"] == 1]
recovered = d["total_rec_prncp"].fillna(0) + d["recoveries"].fillna(0)
lgd = ((d["loan_amnt"] - recovered) / d["loan_amnt"]).clip(0, 1).mean()
```

→ **0.6221**. Estimated from realized charge-offs, so it is an input to the *cost
model*, never a feature (it is unavailable for a live application).

The consequence is the counterintuitive number in finding 3: with mean forgone interest
of \$4,274 and mean `cost_fn` around \$9,000, the **FN:FP ratio is only ≈2.1:1** — not
the 10:1 lending intuition suggests. These are high-rate unsecured loans; turning away
a good borrower forfeits a lot of interest.

### 7.3 The cost curve (`evaluate.cost_curve`)

Decision rule: **approve when p < threshold**. Naive evaluation would be
O(thresholds × loans); this is O(n log n) once, via sorting and prefix sums:

```python
order = np.argsort(p); p_s, y_s = p[order], y[order]

cum_fn     = concat([[0], cumsum(where(y_s, fn_s, 0))])          # approved defaults
rev        = cumsum(where(~y_s, fp_s, 0)[::-1])[::-1]            # declined goods
cum_fp_tail= concat([rev, [0]])

cut   = np.searchsorted(p_s, grid, side="left")   # how many loans fall below threshold
total = cum_fn[cut] + cum_fp_tail[cut]
```

`cut` is the number of approvals. `cum_fn[cut]` sums the FN cost of everything approved;
`cum_fp_tail[cut]` sums the FP cost of everything declined. A threshold of 1.0 approves
everyone — that is the **approve-all baseline**, and every saving figure is quoted
against it, not against p = 0.5.

Note this is a **realized** loss on the labeled test set, not an expected value under
the model. That is the stronger claim: it's what the policy would actually have cost.

### 7.4 Choosing the threshold (`evaluate.select_threshold`)

The function returns the argmin **and** the flat region:

```python
near = curve[curve["total_cost"] <= best["total_cost"] * (1 + tolerance)]   # tolerance = 1%
flat_region = [near["threshold"].min(), near["threshold"].max()]
```

On this data the band is **0.41–0.61** — cost varies by under 1% across it. So the
validation argmin (0.44) and test argmin (0.48) differ by pure sampling noise, and
quoting either as *the* threshold is false precision.

`train_calibrate.py` therefore serves the **midpoint of the band, 0.51**:

```python
lo, hi = chosen["flat_region"]
t = round((lo + hi) / 2, 3)
```

Selected on validation, applied unchanged to test. Test result at 0.51: **96.4%
approval, \$1,893 cost/application, \$9.37M saved vs approve-all** over 201,803 loans.

`evaluate.lgd_sensitivity()` then re-runs the whole search across LGD ∈ {0.4, 0.5,
0.622, 0.7, 0.8}. Threshold swings 0.62 → 0.36 and savings swing \$0.6M → \$34.3M.
**The LGD assumption dominates the model.** That table is the most decision-relevant
output in the repo.

---

## 8. The six experiments

### 8.1 `run_baselines.py` — how much lift is real

Fits `grade_only` (logistic — one feature needs no tree), `incumbent_only`,
`applicant_only`, `full` (LightGBM), plus `full_logreg`, all on the shared split.

| Feature set | n | AUC | Δ vs grade-only |
|---|---|---|---|
| grade_only | 1 | 0.6794 | — |
| incumbent_only | 3 | 0.6968 | +0.0174 |
| **applicant_only** | 43 | **0.7182** | **+0.0388** |
| full | 46 | 0.7243 | +0.0449 |
| full_logreg | 46 | 0.7101 | +0.0307 |

Two readings, and the second is the one people miss:

- Borrower attributes alone (0.7182) **beat the entire incumbent scorecard** (0.6968) →
  the model substitutes for the grading system rather than echoing it.
- Adding `grade`/`int_rate` on top of borrower attributes buys only **+0.006** → the
  incumbent scorecard carries almost no information the raw application lacks.

### 8.2 `train_calibrate.py` — probabilities, not scores

Fits the same architecture twice, with and without `scale_pos_weight = 4.01`:

| Variant | AUC | Brier | ECE | Mean predicted | Observed |
|---|---|---|---|---|---|
| weighted | 0.7077 | 0.1557 | 0.0759 | 25.1% | 20.0% |
| unweighted | 0.7243 | 0.1427 | 0.0020 | 20.0% | 20.0% |
| unweighted + isotonic | 0.7242 | 0.1428 | 0.0019 | 20.0% | 20.0% |

`scale_pos_weight` is a **ranking** device — it re-weights the loss so the minority
class dominates, which deliberately shifts predicted probabilities upward. The old
Streamlit app displayed that output as "probability of default," inflating shown risk
by roughly 3×.

The honest finding: **isotonic regression was not needed.** LightGBM minimizing log-loss
is already calibrated to ECE 0.0020. The entire miscalibration was self-inflicted by the
weighting — which *also* cost 0.017 AUC. The calibrator is retained in the serving path
because it is near-identity here and provides a hook if the base rate shifts, but
claiming it fixed the problem would be a lie; removing `scale_pos_weight` did.

The isotonic fit uses **validation** predictions and is scored on **test**, so the
calibration numbers are out-of-sample for the calibrator.

**ECE implementation** (`evaluate.expected_calibration_error`): 20 equal-width bins,
weighted by bin population, |mean predicted − observed| per bin. See §10.4 for why
equal-width binning flatters the low-base-rate horizon target.

### 8.3 `temporal_validation.py` — the drop, and the confound

Train ≤2014 → validate 2015 → test 2016–18. AUC 0.7359 → 0.7121, **drop 0.0238**.

Rather than call that decay, the script isolates the mechanism. 2014 and 2015 contain
*both* matured and un-matured resolved loans, so the same model scored on each subset
holds the origination year fixed:

| Year | Matured | n | Default rate | AUC |
|---|---|---|---|---|
| 2014 | no | 60,533 | 31.1% | 0.6950 |
| 2014 | yes | 162,570 | 13.7% | 0.7230 |
| 2015 | no | 92,520 | 36.4% | 0.6751 |
| 2015 | yes | 283,026 | 14.9% | 0.6970 |

Un-matured loans default at **2.3×** the rate and score **0.0250 AUC worse** — inside
the same origination years, where no distribution shift exists by construction. Since
the 2016+ test set is 100% un-matured, that 0.0250 penalty alone is the size of the
observed 0.0238 drop.

Conclusion drawn at this stage: the decay is survivorship. **§8.4 partially overturns
this.**

### 8.4 `horizon_validation.py` — the direct test

§8.3 argued from a proxy. This script removes the bias at the source by re-running the
identical out-of-time protocol on target B.

| Target | In-time AUC | Out-of-time AUC | Drop |
|---|---|---|---|
| resolved-only | 0.7359 | 0.7121 | **0.0238** |
| 12-month window | 0.7249 | 0.7119 | **0.0131** |

**This contradicts §8.3, and the repo says so.** Roughly half the apparent degradation
was survivorship; a residual 0.0131 survives the correction, so there is mild *genuine*
drift. `reports/horizon_per_year.csv` agrees — AUC 0.794 (2014) → 0.725 (2015) → 0.715
(2016) → 0.709 (2017).

The vintage curve also inverts the narrative entirely:

| Issue year | Default rate, resolved-only | Default rate, 12-month window |
|---|---|---|
| 2013 | 15.6% | 3.2% |
| 2014 | 18.5% | 3.4% |
| 2015 | 20.2% | 4.0% |
| 2016 | 23.3% | 4.7% |
| 2017 | 23.1% | 4.4% |

The resolved-only column looks like credit quality collapsing after 2015. It isn't —
that column is measuring **how much of each vintage had resolved**, not how much had
defaulted.

### 8.5 `leakage_demo.py` — why published 0.95s are meaningless

Same split, same model, plus `config.LEAKY`:

| Column | Why it's post-origination |
|---|---|
| `last_fico_range_low/high` | credit score at the *most recent* pull — for a defaulted loan, pulled after the default |
| `last_pymnt_amnt` | size of the final payment received |
| `out_prncp` | outstanding principal — 0 for every resolved loan, so it encodes resolution itself |
| `total_pymnt`, `total_rec_int` | cumulative amounts actually collected |

| Feature set | Test AUC | AP | Deployable |
|---|---|---|---|
| full | 0.7243 | 0.3943 | yes |
| full_leaky | **0.9995** | 0.9987 | **no** |

The script also dumps top-10 feature importance and flags which entries are leaked:
`total_pymnt`, `total_rec_int`, `last_pymnt_amnt`, `last_fico_range_low` all rank in the
top 10. `total_pymnt` dominates, which is unsurprising — a fully repaid loan returns
principal plus interest and a charged-off one does not, so the column is very nearly the
label arithmetic.

This is a runnable script rather than a README caveat because *"our 0.72 is honest and
their 0.95 is leakage"* is a claim that should be demonstrable on demand.

### 8.6 `fairness_audit.py` — geographic proxy disparity screen

The model never consumes `addr_state` or masked `zip_code`; `config.FAIRNESS_PROXIES`
keeps them available only for evaluation. The audit scores the held-out test split with
the shipped model and reports approval rate, default rate, good-applicant approval
rate, bad-loan approval rate, and each group's approval rate relative to the
highest-approval group. Groups under 0.80 are flagged for investigation.

On the 201,803-loan test set, 43 state groups and 110 masked-ZIP groups meet the
500-loan reporting floor. Their minimum ratios are 0.961 and 0.956 respectively;
none fall below 0.80. The portfolio approves 96.39% of applications at the selected
threshold, so this is a low-rejection diagnostic rather than a strong fairness stress
test.

This is a diagnostic, not a legal conclusion. The extract has no complete protected-
class labels, geography is only a proxy, and excluding geography cannot remove all
correlated information. `MODEL_CARD.md` states that limitation explicitly.

---

## 9. The serving path

### 9.1 `FeatureSpec` — the train/serve contract

Four artifacts are written by `train_calibrate.py` and loaded by the app:

```
processed/lgbm_model.pkl        unweighted LightGBM
processed/calibrator.pkl        fitted IsotonicRegression
processed/feature_spec.json     {name, numeric, categorical, columns}
processed/serving_config.json   {threshold, trained_on}
```

`feature_spec.json` freezes the **exact fitted column list, in order**. Both training
and serving call the same method:

```python
def transform(self, df):
    if "loan_to_income" not in df.columns:
        df = engineer(df)                       # idempotent
    out = self._encode(df)                      # numeric + get_dummies(categorical)
    return out.reindex(columns=self.columns, fill_value=0.0).astype(float)
```

The `reindex` does two jobs at once: it adds dummy levels absent from a single-row input
(a lone applicant produces exactly one `purpose_*` column) and enforces column order.

**The bug this replaced:** the app previously retyped the one-hot column list by hand.
It appeared to work only because any name that didn't match was silently zero-filled —
so a typo degraded predictions invisibly rather than raising. Deriving the list from the
fitted artifact makes that class of bug impossible.

### 9.2 Two live bugs fixed in the app

1. **Uncalibrated scores shown as probabilities.** The app rendered the raw
   `scale_pos_weight`-inflated output as a percentage — roughly 3× the true rate. Now it
   serves the unweighted model through the isotonic calibrator.
2. **SHAP explaining the wrong class.** The waterfall used `explainer.expected_value[0]`,
   which for binary LightGBM is the **repayment** class. Every contribution was shown
   with the sign flipped — a feature that *raised* default risk was drawn as lowering it.
   Now:
   ```python
   if isinstance(shap_values, list):
       sv, base = shap_values[1][0], explainer.expected_value[1]   # index 1 = default
   else:
       sv, base = shap_values[0], explainer.expected_value
   ```
   The list-vs-array branch is required because SHAP's return shape for binary LightGBM
   varies by version.

The app also derives `installment` from the standard amortization formula rather than
asking for it, so the user cannot enter a payment inconsistent with amount/rate/term.

The SHAP caption states explicitly that contributions are on the **raw log-odds scale,
before calibration** — directionally consistent with the displayed probability but not
on the same scale. Worth keeping; it is a real limitation of explaining a calibrated
model with a tree explainer.

---

## 10. Known soft spots

Ordered by how much they'd matter if this were real.

### 10.1 The deployed model uses the biased target ⚠️

`train_calibrate.py` reads `loans.parquet` — **target A**. So the shipped model, its
calibration, its 0.51 threshold, and every dollar figure in `decision.json` rest on the
resolved-only label that §8.3–8.4 showed is survivorship-biased for 2016+. The random
split spreads that bias across train/val/test evenly, so the *internal* comparisons stay
valid, but the absolute 20% base rate the app calibrates to is inflated by early-resolver
selection.

The natural next step is retraining the production path on target B. That is not a
drop-in change: `build_horizon()` attaches neither `matured` nor the economics columns
(`forgone_interest`, `cost_fn`, `cost_fp`), so the cost model would need wiring in, and
the whole threshold search would have to be redone against a ~4% base rate.

### 10.2 The residual 0.0131 drift is unexplained

§8.4 removes survivorship but leaves real degradation. Two untested candidates:
LendingClub's borrower mix widening after 2015, or the newer bureau fields (`bc_util`,
`tot_cur_bal`, `acc_open_past_24mths`) being absent before 2012 — which makes early
vintages structurally different in ways unrelated to time. Not yet separated.

### 10.3 The validation set does triple duty

`val` is used for (a) LightGBM early stopping, (b) fitting the isotonic calibrator, and
(c) selecting the threshold. Each use leaks a little of `val` into the fitted object.
All headline numbers are reported on `test`, which the calibrator and threshold never
saw, so the reported figures are sound — but the *validation* metrics are optimistic and
should not be quoted.

### 10.4 ECE uses equal-width bins

`expected_calibration_error` splits [0, 1] into 20 uniform bins. For target A (mean
prediction 0.20) that resolves fine. For target B (mean prediction 0.033) nearly all
mass lands in the first one or two bins, so the reported ECE of 0.008 is **flattered by
the binning**, not necessarily by the model. Equal-frequency (quantile) bins would be
the honest comparison for low-base-rate targets.

### 10.5 `flat_region` assumes contiguity

`select_threshold` reports `[min, max]` of all thresholds within 1% of the optimum. If
the near-optimal set were non-contiguous, that interval would overstate it. On this data
the curve is unimodal so it's fine, but the function does not verify contiguity.

### 10.6 Cost-model simplifications

- **`cost_fp` treats a declined applicant as a total loss of scheduled interest.** In
  practice they'd more likely be re-priced than rejected outright, so FP cost is an
  upper bound — which biases the threshold *high* (toward over-approving).
- **No discounting.** Interest earned in month 60 is counted at face value alongside
  principal lost in month 3.
- **LGD is a single portfolio average.** It almost certainly varies by grade and term. A
  per-segment estimate would sharpen the decision more than any modeling change — see
  the sensitivity table in §7.4.

### 10.7 Fairness data is proxy-only

The reproducible audit covers state and masked ZIP, and both are excluded from model
features. Those fields are geographic proxies rather than protected-class labels, so
the results cannot establish demographic parity, intersectional fairness, or legal
compliance. A real deployment still requires a controlled protected-class analysis.

### 10.8 12-month horizon truncation

Target B excludes 2018 originations entirely (no full window before the snapshot) and
truncates defaults occurring in months 13+. A 24-month window would capture more of the
realized loss but cover fewer vintages — a direct bias/coverage trade-off that was
resolved in favor of coverage.

---

## 11. Reproducing

```bash
pip install -r requirements.txt   # versions pinned; results produced on Python 3.10

# accepted_2007_to_2018Q4.csv (~1.6 GB) must sit in the repo root:
# https://www.kaggle.com/datasets/wordsforthewise/lending-club

python scripts/prepare_data.py         # ~4 min; writes both parquet tables
python scripts/run_baselines.py        # §8.1
python scripts/train_calibrate.py      # §8.2, §8.3 economics; writes serving artifacts
python scripts/temporal_validation.py  # §8.3
python scripts/horizon_validation.py   # §8.4 — requires temporal.json
python scripts/leakage_demo.py         # §8.5
python scripts/fairness_audit.py        # §8.6

pip install -r requirements-dev.txt
ruff check src scripts tests streamlit_app.py
pytest

streamlit run streamlit_app.py
```

Seed is fixed at 42 throughout (`config.RANDOM_SEED`) and every reported number is
regenerated from these seven commands. `reports/` is tracked in git so results are
diffable across runs.

---

## 12. Interview quick-reference

Likely questions and where the answer lives:

| Question | Section |
|---|---|
| "0.72 AUC is low — is this model any good?" | §8.1 — it beats the incumbent scorecard; and §8.5 for why higher numbers are fake |
| "Why not just use `scale_pos_weight` for imbalance?" | §8.2 — it's a ranking device, cost 0.017 AUC and 38× the ECE |
| "How did you pick the threshold?" | §7.3–7.4 — expected loss with per-loan costs, reported as a band |
| "How do you know it's survivorship and not drift?" | §8.3 then §8.4 — and it's *both*, roughly half each |
| "Isn't `last_pymnt_d` leakage?" | §3.2 — label construction, never a feature |
| "What would you do next?" | §10.1 (retrain on target B), §10.6 (per-segment LGD), §10.7 (fairness) |
