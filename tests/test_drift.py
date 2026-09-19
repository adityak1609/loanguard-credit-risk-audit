from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loanguard.drift import categorical_psi, feature_shift_report, numeric_psi


def test_numeric_psi_is_zero_for_identical_populations():
    values = np.arange(100)
    assert numeric_psi(values, values) == pytest.approx(0.0)


def test_numeric_psi_detects_distribution_shift():
    reference = np.arange(100)
    shifted = np.arange(100) + 100
    assert numeric_psi(reference, shifted) > 1.0


def test_numeric_psi_validates_bin_count():
    with pytest.raises(ValueError, match="at least 2"):
        numeric_psi([1, 2], [1, 2], n_bins=1)


def test_categorical_psi_detects_mix_shift_and_missing_values():
    reference = ["A"] * 80 + ["B"] * 20 + [None] * 10
    current = ["A"] * 20 + ["B"] * 80 + [None] * 10
    assert categorical_psi(reference, current) > 1.0


def test_feature_shift_report_is_sorted_and_tracks_missingness():
    reference = pd.DataFrame({"stable": [1, 2, 3, 4], "shifted": ["A"] * 4})
    current = pd.DataFrame(
        {"stable": [1, 2, 3, 4], "shifted": ["B", "B", "B", None]}
    )
    report = feature_shift_report(reference, current, ["stable"], ["shifted"])
    assert report.iloc[0]["feature"] == "shifted"
    assert report.iloc[0]["missing_rate_change"] == pytest.approx(0.25)
