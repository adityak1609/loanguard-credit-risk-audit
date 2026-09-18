from __future__ import annotations

import pandas as pd
import pytest

from loanguard import config as C
from loanguard.data import add_costs, build, build_horizon, estimate_lgd


def test_resolved_target_excludes_unresolved_loans(raw_loans):
    result = build(raw_loans)
    assert len(result) == 2
    assert result["target"].tolist() == [0, 1]


def test_data_coercion_is_shared_with_target_build(raw_loans):
    result = build(raw_loans)
    assert result["grade"].tolist() == [1, 4]
    assert result["sub_grade"].tolist() == [6, 23]
    assert result["term_months"].tolist() == [36, 36]
    assert result["int_rate"].tolist() == [10.0, 20.0]


def test_fixed_horizon_keeps_current_loans_as_negatives(raw_loans):
    result = build_horizon(raw_loans, horizon=12, dpd_lag=3)
    current = result[result["loan_status"] == "Current"].iloc[0]
    assert current["target"] == 0


def test_fixed_horizon_dates_default_from_last_payment(raw_loans):
    result = build_horizon(raw_loans, horizon=12, dpd_lag=3)
    charged_off = result[result["loan_status"] == "Charged Off"].iloc[0]
    assert charged_off["target"] == 1


def test_estimate_lgd_uses_recovery_and_principal(raw_loans):
    result = build(raw_loans)
    assert estimate_lgd(result) == pytest.approx(0.5)


def test_estimate_lgd_falls_back_when_no_defaults(raw_loans):
    result = build(raw_loans.iloc[[0]])
    assert estimate_lgd(result) == C.FALLBACK_LGD


def test_add_costs_does_not_mutate_input():
    frame = pd.DataFrame({"loan_amnt": [1_000.0], "forgone_interest": [100.0]})
    result = add_costs(frame, 0.6)
    assert "cost_fn" not in frame
    assert result.loc[0, "cost_fn"] == 600.0
    assert result.loc[0, "cost_fp"] == 100.0
