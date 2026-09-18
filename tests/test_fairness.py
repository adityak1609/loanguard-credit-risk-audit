from __future__ import annotations

import pandas as pd
import pytest

from loanguard.fairness import audit_summary, group_report


def test_group_report_computes_four_fifths_ratio():
    report = group_report(
        group=["A"] * 10 + ["B"] * 10,
        approved=[True] * 10 + [True] * 7 + [False] * 3,
        min_group_size=1,
    ).set_index("group")
    assert report.loc["A", "adverse_impact_ratio"] == 1.0
    assert report.loc["B", "adverse_impact_ratio"] == pytest.approx(0.7)
    assert bool(report.loc["B", "below_four_fifths"])


def test_group_report_excludes_small_groups():
    report = group_report(["large"] * 5 + ["small"], [True] * 6, min_group_size=2)
    assert report["group"].tolist() == ["large"]


def test_group_report_includes_outcome_conditioned_metrics():
    report = group_report(
        ["A"] * 4,
        [True, False, True, False],
        target=[0, 0, 1, 1],
        score=[0.1, 0.6, 0.4, 0.8],
        min_group_size=1,
    ).iloc[0]
    assert report["good_approval_rate"] == 0.5
    assert report["bad_approval_rate"] == 0.5
    assert report["observed_default_rate"] == 0.5
    assert report["mean_predicted_default"] == pytest.approx(0.475)


def test_group_report_rejects_invalid_minimum_size():
    with pytest.raises(ValueError, match="positive"):
        group_report(["A"], [True], min_group_size=0)


def test_audit_summary_is_json_serialisable_and_lists_flags():
    report = pd.DataFrame(
        {
            "group": ["A", "B"],
            "adverse_impact_ratio": [1.0, 0.7],
            "below_four_fifths": [False, True],
        }
    )
    summary = audit_summary({"state": report})
    assert summary["proxies"]["state"]["groups_below_four_fifths"] == ["B"]
    assert summary["screening_threshold"] == 0.80
