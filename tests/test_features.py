from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loanguard import config as C
from loanguard.features import FEATURE_SETS, FeatureSpec, engineer


def test_engineer_computes_expected_ratios(feature_frame):
    row = feature_frame.iloc[0]
    assert row["loan_to_income"] == pytest.approx(0.2)
    assert row["installment_to_monthly_income"] == pytest.approx(320 / (50_000 / 12))
    assert row["revol_bal_to_income"] == pytest.approx(0.1)


def test_engineer_zero_income_produces_missing_ratios(feature_frame):
    row = feature_frame.iloc[1]
    assert np.isnan(row["loan_to_income"])
    assert np.isnan(row["installment_to_monthly_income"])


def test_engineer_does_not_mutate_input(feature_frame):
    raw = feature_frame.drop(columns=C.ENGINEERED[:-1])
    before = raw.copy(deep=True)
    engineer(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_feature_spec_preserves_fitted_column_order(feature_frame):
    spec = FeatureSpec.build("full", feature_frame)
    transformed = spec.transform(feature_frame.iloc[::-1])
    assert transformed.columns.tolist() == spec.columns


def test_unknown_category_is_not_added_at_serving(feature_frame):
    spec = FeatureSpec.build("full", feature_frame)
    serving = feature_frame.iloc[[0]].copy()
    serving["purpose"] = "never_seen_during_training"
    transformed = spec.transform(serving)
    assert "purpose_never_seen_during_training" not in transformed.columns
    assert transformed.filter(like="purpose_").to_numpy().sum() == 0


def test_missing_training_category_is_zero_filled(feature_frame):
    spec = FeatureSpec.build("full", feature_frame)
    serving = feature_frame[feature_frame["home_ownership"] != "OWN"]
    transformed = spec.transform(serving)
    assert (transformed["home_ownership_OWN"] == 0).all()


def test_saved_spec_round_trip_guarantees_train_serve_parity(feature_frame, tmp_path):
    spec = FeatureSpec.build("full", feature_frame)
    path = tmp_path / "feature_spec.json"
    spec.save(path)
    loaded = FeatureSpec.load(path)
    pd.testing.assert_frame_equal(spec.transform(feature_frame), loaded.transform(feature_frame))


def test_geographic_proxies_are_excluded_from_deployable_feature_sets():
    for name, (numeric, categorical) in FEATURE_SETS.items():
        if name == "full_leaky":
            continue
        assert set(C.FAIRNESS_PROXIES).isdisjoint(numeric + categorical)


def test_post_origination_columns_only_exist_in_explicit_leakage_set():
    full_columns = set(sum(FEATURE_SETS["full"], []))
    leaky_columns = set(sum(FEATURE_SETS["full_leaky"], []))
    assert set(C.LEAKY).isdisjoint(full_columns)
    assert set(C.LEAKY).issubset(leaky_columns)
