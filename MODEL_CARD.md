# Model Card: LoanGuard Credit-Risk Model Audit

## Model details

LoanGuard is an audit-first credit-risk case study built on the public
LendingClub 2007–2018 accepted-loan extract. The served estimator is an
unweighted LightGBM binary classifier followed by isotonic calibration. Its
decision threshold is selected on a validation set using estimated portfolio
costs, not classification accuracy or an arbitrary 0.5 cutoff.

- Owner: Aditya Khanna
- Version: repository `main` branch
- License: MIT
- Training code: `scripts/train_calibrate.py`
- Random seed: 42

## Intended use

This project is intended for education, portfolio review, and reproducible
analysis of leakage, probability calibration, temporal validation, and
decision thresholds. It demonstrates how apparently strong credit-model
results can be misleading.

It is **not intended for real lending decisions**, pricing, adverse-action
notices, or regulatory compliance. The source data is historical, the target
has known survivorship limitations, protected-class labels are unavailable,
and no production governance process is included.

## Training and evaluation data

- Source: LendingClub accepted loans, 2007–2018Q4.
- Served-model target: terminal status among 1,345,350 resolved loans.
- Alternative research target: default within a fixed 12-month observation
  window among 1,765,426 loans.
- Random model split: 70% train, 15% validation, 15% test, stratified.
- Temporal evaluation: train through 2014, validate on 2015, test on 2016–2017
  for the fixed-window target.

The raw CSV is not redistributed. Preparation is deterministic through
`scripts/prepare_data.py`.

## Inputs and exclusions

The model uses application-time borrower, bureau, loan, and LendingClub
scorecard fields plus six deterministic engineered features. The exact fitted
schema and column order are frozen in `processed/feature_spec.json` and shared
with the serving application.

The following are explicitly excluded from the deployable model:

- Post-origination outcome fields such as `total_pymnt`, `out_prncp`, and
  `last_pymnt_amnt`, because they leak the target.
- `addr_state` and `zip_code`, because geographic fields can act as proxies for
  protected characteristics. They are retained only for monitoring.

## Performance

On the held-out random test split:

| Metric | Value |
|---|---:|
| ROC AUC | 0.7242 |
| Average precision | 0.3943 |
| Brier score | 0.1428 |
| Expected calibration error | 0.0019 |
| Observed default rate | 19.97% |
| Approval rate at the selected threshold | 96.4% |

The cost-optimal region is broad: thresholds from roughly 0.41 to 0.61 are
within 1% of the minimum estimated loss. The served value (0.51) is the
midpoint, not a precisely identified optimum.

For the fixed 12-month target, AUC falls from 0.7249 on 2015 to 0.7119 on
2016–2017. Roughly half of the larger drop seen with the resolved-only target
is attributable to survivorship bias; a residual 0.0131 remains. Retraining on
2015 raises 2016–2017 AUC to 0.7205, recovering 65.7% of that residual loss.
The drop is concentrated in applicant features and occurs within every analyzed
grade and term segment, which is consistent with model staleness rather than a
simple change in portfolio mix.

## Fairness and subgroup evaluation

The dataset contains no direct race, ethnicity, sex, age, or other complete
protected-class labels. A reproducible proxy audit therefore reports approval
rates by state and masked ZIP code, with the four-fifths ratio used only as a
screening signal. Run `python scripts/fairness_audit.py` to generate:

- `reports/fairness_addr_state.csv`
- `reports/fairness_zip_code.csv`
- `reports/fairness_summary.json`

On the 201,803-loan held-out test set, the overall approval rate is 96.39%.
Among groups with at least 500 observations, the minimum approval-rate ratio is
0.961 across 43 states and 0.956 across 110 masked ZIP groups. No group falls
below the 0.80 screening threshold. Because declines are uncommon at this
operating point, this result should not be read as evidence that stricter
thresholds would remain disparity-free.

This geographic analysis cannot establish demographic fairness or legal
compliance. Excluding geography also does not remove proxy information already
present in correlated financial variables. A real deployment would require
lawful collection or a controlled third-party analysis of protected-class
outcomes, intersectional testing, uncertainty intervals, and ongoing drift
monitoring.

## Known limitations and risks

- The served model uses the resolved-only target, which is biased for late
  vintages; the fixed-window target is the better research design.
- LendingClub grade and interest rate are outputs of an incumbent scorecard.
  Results are therefore also reported without them.
- Loss given default is a portfolio average and materially changes the chosen
  operating point.
- The 12-month event date uses `last_pymnt_d + 3 months` as a delinquency proxy.
- Historical accepted-loan data omits rejected applicants, creating selection
  bias and preventing conclusions about the full applicant population.
- Calibration and ranking performance may not transfer to another lender,
  product, time period, or macroeconomic regime.
- SHAP values explain the model, not causality or the appropriateness of a
  lending decision.

## Reproducibility and tests

`pytest` covers feature engineering, serialized `FeatureSpec` parity,
post-origination leakage boundaries, target construction, calibration,
cost-threshold selection, drift diagnostics, and fairness-report calculations. GitHub Actions
runs linting, tests, and entry-point compilation on every push and pull request.

The numerical reports tracked under `reports/` are the reproducibility
artifacts for the claims in the README.
