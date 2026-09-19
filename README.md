# LoanGuard: A Credit-Risk Model Audit

[![CI](https://github.com/adityak1609/loanguard-credit-risk-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/adityak1609/loanguard-credit-risk-audit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**[Try the live app](https://loan-d-qwshwdrewxtufmpqexbqd3.streamlit.app/)** ·
**[Read the model card](MODEL_CARD.md)** ·
**[See the technical details](TECHNICAL.md)**

LoanGuard is an audit of a credit-default model trained on more than 1.35 million
LendingClub loans. The goal is not simply to produce a high score. It is to check
whether the score is honest, usable, and reproducible.

## Results in 10 seconds

On the 201,803-loan test set, the cost-based policy approves 96.4% of applications
and reduces estimated loss by **$9.37 million compared with approving everyone**.
That estimate depends on the loss assumptions described below; it is not a claim
of real-world profit.

| Result | Plain-English meaning |
|---|---|
| Honest test AUC: **0.724** | The model ranks risky loans reasonably well; 0.5 is random and 1.0 is perfect. |
| Leaky test AUC: **0.9995** | Near-perfect public results on this dataset can come from using information recorded after the loan outcome. |
| Calibration error: **0.002** | Predicted default probabilities closely match observed default rates. |
| Estimated savings: **$9.37M** | The selected threshold lowers estimated loss versus approving every test application. |
| Corrected temporal drop: **0.013 AUC** | About half of the apparent performance decline was caused by biased sampling; a smaller real decline remains. |
| Geographic proxy audit | No state or masked-ZIP group crossed the four-fifths screening threshold. |

## What the audit found

### The original probabilities were too high

Class weighting made the model's risk estimates look roughly three times larger
than they should have. Removing the weighting fixed most of the problem. Isotonic
calibration is retained as a final safety layer.

### AUC above 0.95 was leakage, not better modeling

Fields such as total payments, remaining principal, and last payment amount are
recorded after a loan starts performing. Adding them raises AUC from **0.724 to
0.9995**, but makes the model impossible to use for a new application.

### The first temporal test was biased

Later loans had less time to reach a final status. Filtering only to completed or
defaulted loans therefore over-selected early defaults. A fixed 12-month observation
window recovered more than 400,000 loans and reduced the apparent AUC decline from
**0.024 to 0.013**.

A follow-up test found that refreshing the model on 2015 data raised its 2016–2017
AUC from **0.712 to 0.720**, recovering about two-thirds of the remaining decline.
The evidence points to model staleness in borrower signals rather than missing bureau
fields or a simple change in grade mix.

### The decision threshold is an economic choice

The best threshold is not automatically 0.5. LoanGuard chooses it by comparing the
estimated cost of approving a default with the interest lost by rejecting a good
loan. The near-optimal range is broad, so the project reports a range instead of
pretending one exact cutoff is certain.

### Geography is monitored, not used for prediction

`addr_state` and masked `zip_code` are excluded from the model. They are used only
for a lightweight disparity screen. This is useful monitoring, but it is not a legal
fairness assessment because the dataset has no protected-class labels.

## What makes this project different

- Training and serving use the same saved `FeatureSpec`, preventing silent column
  or category mismatches.
- Post-origination fields are explicitly separated and tested as leakage.
- Evaluation covers ranking, probability calibration, financial cost, and time drift.
- The repository includes automated tests, linting, CI, a model card, and tracked
  result files.
- The README reports negative and contradictory findings instead of hiding them.

## Run the app

The trained artifacts are included, so the demo does not require the raw dataset.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Reproduce the analysis

Download `accepted_2007_to_2018Q4.csv` from the
[LendingClub dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
and place it in the repository root.

```bash
python scripts/prepare_data.py
python scripts/run_baselines.py
python scripts/train_calibrate.py
python scripts/temporal_validation.py
python scripts/horizon_validation.py
python scripts/drift_diagnosis.py
python scripts/leakage_demo.py
python scripts/fairness_audit.py
```

All experiments use seed `42`. Generated summaries are stored in `reports/`.

## Tests and CI

```bash
pip install -r requirements-dev.txt
pytest
ruff check src scripts tests streamlit_app.py
```

The test suite covers feature engineering, train/serve parity, leakage guards,
target construction, calibration, cost-threshold math, drift utilities, and fairness
reporting. GitHub Actions runs tests and linting on every push and pull request.

## Repository guide

| Path | Purpose |
|---|---|
| `src/loanguard/` | Reusable data, feature, model, evaluation, drift, and fairness code |
| `scripts/` | Reproducible experiment entry points |
| `tests/` | Unit tests for the main contracts and calculations |
| `reports/` | Tracked outputs supporting the reported results |
| `processed/` | Small serving artifacts; large Parquet files are ignored |
| `MODEL_CARD.md` | Intended use, limitations, performance, and fairness notes |
| `TECHNICAL.md` | Full methodology and design decisions |

## Important limitations

- This is a portfolio and research project, not a production underwriting system.
- The served model still uses a resolved-loan target with known survivorship bias;
  the fixed-window target is used to study that limitation.
- The $9.37M estimate changes when loss-given-default or rejection-cost assumptions
  change.
- The fairness check uses geography as a proxy and cannot establish demographic or
  regulatory fairness.
- Historical accepted loans do not represent applicants LendingClub rejected.

See [MODEL_CARD.md](MODEL_CARD.md) for the complete use boundary and
[TECHNICAL.md](TECHNICAL.md) for the full analysis.

MIT licensed. See [LICENSE](LICENSE).
