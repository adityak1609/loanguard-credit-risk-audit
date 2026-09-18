"""Paths, column groups, and economic assumptions for LoanGuard."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "accepted_2007_to_2018Q4.csv"
PROCESSED = ROOT / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

RANDOM_SEED = 42

# The dataset snapshot ends at 2018Q4. Any loan whose term extends past this
# date cannot have reached a terminal status yet, which is the source of the
# maturity bias described in the README.
SNAPSHOT_END = "2018-12-31"

# ── Target ───────────────────────────────────────────────────────────────
TERMINAL_STATUSES = ["Fully Paid", "Charged Off", "Default"]
DEFAULT_STATUSES = ["Charged Off", "Default"]

# ── Fixed observation window ─────────────────────────────────────────────
# The terminal-status target above is only observable for loans that have
# resolved, which biases late vintages toward early defaulters. Labelling
# "defaulted within HORIZON_MONTHS of origination" instead makes every vintage
# comparable and lets still-performing loans count as negatives.
HORIZON_MONTHS = 12

# Statuses that count as a default event. In Grace Period and Late (16-30
# days) are excluded: both are under 31 days delinquent and frequently cure.
DEFAULT_EVENT_STATUSES = [
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
]

# `last_pymnt_d` dates the last payment received, i.e. when the borrower
# stopped paying. Reaching 90+ days delinquent follows roughly this many
# months later, so the delinquency event is dated last_pymnt_d + lag.
DPD_LAG_MONTHS = 3

# ── Post-origination columns ─────────────────────────────────────────────
# None of these are known when the lending decision is made. `last_fico_*` is
# the borrower's score at the most recent credit pull, which for a defaulted
# loan is measured *after* the default. They exist here only so that
# scripts/leakage_demo.py can show what including them does.
LEAKY = [
    "last_fico_range_low",
    "last_fico_range_high",
    "last_pymnt_amnt",
    "out_prncp",
    "total_pymnt",
    "total_rec_int",
]

# ── Feature groups ───────────────────────────────────────────────────────
# INCUMBENT features are LendingClub's *own* risk model output, not borrower
# attributes. A model trained on them is largely re-learning the existing
# scorecard, so we always report metrics with and without them.
INCUMBENT = ["grade", "sub_grade", "int_rate"]

# Borrower/application attributes known at decision time.
APPLICANT_NUMERIC = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "emp_length",
    "fico_range_low",
    "open_acc",
    "revol_util",
    "revol_bal",
    "mort_acc",
    "pub_rec",
    "total_acc",
    "delinq_2yrs",
    "inq_last_6mths",
    "installment",
]
APPLICANT_CATEGORICAL = ["home_ownership", "purpose", "verification_status"]

# Geographic fields are retained only for post-model fairness monitoring. They
# are deliberately excluded from every deployable feature set: both can act as
# proxies for protected characteristics and neither is needed to reproduce the
# reported model performance.
FAIRNESS_PROXIES = ["addr_state", "zip_code"]

# Derived in features.py
ENGINEERED = [
    "loan_to_income",
    "installment_to_monthly_income",
    "revol_bal_to_income",
    "log_annual_inc",
    "credit_history_years",
    "term_months",
]

# Columns needed only for bookkeeping / economics, never fed to the model.
META = [
    "issue_d",
    "term",
    "loan_status",
    "earliest_cr_line",
    "last_pymnt_d",
    "total_rec_prncp",
    "recoveries",
]

USECOLS = sorted(
    set(
        INCUMBENT
        + APPLICANT_NUMERIC
        + APPLICANT_CATEGORICAL
        + FAIRNESS_PROXIES
        + META
        + LEAKY
    )
)

# ── Economics ────────────────────────────────────────────────────────────
# Loss given default is estimated empirically from charged-off loans in
# estimate_lgd(); this is only the fallback if that estimate is unavailable.
FALLBACK_LGD = 0.70

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}
GRADE_MAP = {g: i + 1 for i, g in enumerate("ABCDEFG")}
