from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loanguard.evaluate import (
    calibration_metrics,
    confusion_at,
    cost_curve,
    expected_calibration_error,
    ranking_metrics,
    reliability_curve,
    select_threshold,
)


def test_ranking_metrics_for_perfect_ranking():
    metrics = ranking_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["base_rate"] == 0.5


def test_calibration_metrics_for_exact_binary_probabilities():
    metrics = calibration_metrics(np.array([0, 1]), np.array([0.0, 1.0]))
    assert metrics["brier"] == 0.0
    assert metrics["ece"] == 0.0


def test_ece_weights_bins_by_population():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.1, 0.6, 0.6])
    assert expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.25)


def test_reliability_curve_omits_empty_bins():
    curve = reliability_curve(np.array([0, 1]), np.array([0.1, 0.9]), n_bins=10)
    assert len(curve) == 2
    assert curve["n"].sum() == 2


def test_cost_curve_matches_hand_calculation():
    curve = cost_curve(
        y=np.array([1, 0]),
        p=np.array([0.2, 0.8]),
        cost_fn=np.array([100.0, 100.0]),
        cost_fp=np.array([10.0, 10.0]),
        grid=np.array([0.1, 0.5, 1.0]),
    )
    assert curve["total_cost"].tolist() == [10.0, 110.0, 100.0]
    assert curve["approval_rate"].tolist() == [0.0, 0.5, 1.0]


def test_select_threshold_reports_near_optimal_band():
    curve = pd.DataFrame(
        {
            "threshold": [0.2, 0.3, 0.4, 1.0],
            "total_cost": [101.0, 100.0, 100.5, 150.0],
            "approval_rate": [0.2, 0.3, 0.4, 1.0],
            "cost_per_application": [10.1, 10.0, 10.05, 15.0],
        }
    )
    selected = select_threshold(curve, tolerance=0.01)
    assert selected["threshold"] == 0.3
    assert selected["flat_region"] == [0.2, 0.4]
    assert selected["savings_vs_approve_all"] == 50.0


def test_confusion_at_uses_strict_approval_threshold():
    result = confusion_at([0, 1, 0], [0.1, 0.5, 0.9], threshold=0.5)
    assert result["approved"] == 1
    assert result["declined"] == 2
    assert result["approved_defaults"] == 0
    assert result["declined_goods"] == 1
