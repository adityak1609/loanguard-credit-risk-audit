"""Utilities for diagnosing population shift between loan vintages."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _psi(reference_share: np.ndarray, current_share: np.ndarray) -> float:
    epsilon = 1e-6
    reference_share = np.clip(reference_share, epsilon, None)
    current_share = np.clip(current_share, epsilon, None)
    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def numeric_psi(reference, current, n_bins: int = 10) -> float:
    """Population stability index using reference quantiles plus a missing bin."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    reference = pd.to_numeric(pd.Series(reference), errors="coerce").to_numpy(float)
    current = pd.to_numeric(pd.Series(current), errors="coerce").to_numpy(float)
    ref_finite = reference[np.isfinite(reference)]
    cur_finite = current[np.isfinite(current)]

    if not len(ref_finite):
        return categorical_psi(pd.isna(reference), pd.isna(current))

    quantiles = np.quantile(ref_finite, np.linspace(0, 1, n_bins + 1))
    inner = np.unique(quantiles[1:-1])
    edges = np.concatenate(([-np.inf], inner, [np.inf]))

    ref_counts = np.histogram(ref_finite, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur_finite, bins=edges)[0].astype(float)
    ref_counts = np.append(ref_counts, (~np.isfinite(reference)).sum())
    cur_counts = np.append(cur_counts, (~np.isfinite(current)).sum())
    return _psi(ref_counts / len(reference), cur_counts / len(current))


def categorical_psi(reference, current) -> float:
    """Population stability index across the union of observed categories."""
    reference = pd.Series(reference, dtype="string").fillna("<missing>")
    current = pd.Series(current, dtype="string").fillna("<missing>")
    categories = reference.unique().tolist()
    categories.extend(value for value in current.unique() if value not in categories)
    ref_share = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
    cur_share = current.value_counts(normalize=True).reindex(categories, fill_value=0)
    return _psi(ref_share.to_numpy(), cur_share.to_numpy())


def feature_shift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    """Rank raw model inputs by PSI and expose missingness changes separately."""
    rows = []
    for feature in numeric:
        rows.append(
            {
                "feature": feature,
                "kind": "numeric",
                "psi": numeric_psi(reference[feature], current[feature]),
                "reference_missing_rate": float(reference[feature].isna().mean()),
                "current_missing_rate": float(current[feature].isna().mean()),
            }
        )
    for feature in categorical:
        rows.append(
            {
                "feature": feature,
                "kind": "categorical",
                "psi": categorical_psi(reference[feature], current[feature]),
                "reference_missing_rate": float(reference[feature].isna().mean()),
                "current_missing_rate": float(current[feature].isna().mean()),
            }
        )

    report = pd.DataFrame(rows)
    report["missing_rate_change"] = (
        report["current_missing_rate"] - report["reference_missing_rate"]
    )
    return report.sort_values("psi", ascending=False).reset_index(drop=True)
