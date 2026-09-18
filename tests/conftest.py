from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loanguard import config as C
from loanguard.features import engineer


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    rows = 4
    data = {
        column: np.arange(1, rows + 1, dtype=float)
        for column in C.APPLICANT_NUMERIC + C.INCUMBENT
    }
    data.update(
        {
            "loan_amnt": [10_000, 12_000, 8_000, 20_000],
            "annual_inc": [50_000, 0, 80_000, 100_000],
            "installment": [320, 410, 250, 600],
            "revol_bal": [5_000, 2_000, 8_000, 10_000],
            "issue_d": pd.to_datetime(["2015-01-01"] * rows),
            "earliest_cr_line": pd.to_datetime(
                ["2005-01-01", "2010-01-01", "2000-01-01", "1995-01-01"]
            ),
            "term_months": [36, 36, 60, 60],
            "home_ownership": ["RENT", "OWN", "MORTGAGE", "RENT"],
            "purpose": ["debt_consolidation", "car", "other", "credit_card"],
            "verification_status": [
                "Verified",
                "Not Verified",
                "Source Verified",
                "Verified",
            ],
        }
    )
    return engineer(pd.DataFrame(data))


@pytest.fixture
def raw_loans() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off", "Current"],
            "issue_d": ["Jan-2015", "Jan-2015", "Jan-2017"],
            "last_pymnt_d": ["Jan-2018", "Jun-2015", "Dec-2018"],
            "earliest_cr_line": ["Jan-2005", "Jan-2000", "Jan-2010"],
            "int_rate": ["10.0%", "20.0%", "15.0%"],
            "revol_util": ["30%", "70%", "50%"],
            "emp_length": ["10+ years", "2 years", "< 1 year"],
            "grade": ["A", "D", "C"],
            "sub_grade": ["A1", "D3", "C2"],
            "term": [" 36 months", " 36 months", " 36 months"],
            "loan_amnt": [10_000.0, 10_000.0, 10_000.0],
            "installment": [330.0, 380.0, 350.0],
            "total_rec_prncp": [10_000.0, 4_000.0, 5_000.0],
            "recoveries": [0.0, 1_000.0, 0.0],
        }
    )
