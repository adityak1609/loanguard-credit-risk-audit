"""Group-level monitoring for geographic proxy disparities.

LendingClub does not publish direct protected-class attributes in this extract.
Accordingly, this module does not claim to measure legal or demographic
fairness. It checks whether approval outcomes vary materially across the two
available geographic proxies, which are excluded from the model itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def group_report(
    group,
    approved,
    *,
    target=None,
    score=None,
    min_group_size: int = 500,
) -> pd.DataFrame:
    """Return approval and error-rate metrics for sufficiently large groups.

    The adverse-impact ratio uses the group with the highest approval rate as
    its reference. A value below 0.80 is flagged as a screening diagnostic,
    not as a legal conclusion.
    """
    if min_group_size < 1:
        raise ValueError("min_group_size must be positive")

    frame = pd.DataFrame(
        {
            "group": pd.Series(np.asarray(group), dtype="string").fillna("<missing>"),
            "approved": np.asarray(approved, dtype=bool),
        }
    )
    if target is not None:
        frame["target"] = np.asarray(target, dtype=int)
    if score is not None:
        frame["score"] = np.asarray(score, dtype=float)

    grouped = frame.groupby("group", dropna=False)
    report = grouped.agg(
        n=("approved", "size"),
        approval_rate=("approved", "mean"),
    )
    if score is not None:
        report["mean_predicted_default"] = grouped["score"].mean()
    if target is not None:
        report["observed_default_rate"] = grouped["target"].mean()
        good = frame[frame["target"] == 0].groupby("group")["approved"].agg(
            good_applicants="size", good_approval_rate="mean"
        )
        bad = frame[frame["target"] == 1].groupby("group")["approved"].agg(
            defaulted_applicants="size", bad_approval_rate="mean"
        )
        report = report.join(good).join(bad)

    report = report[report["n"] >= min_group_size].copy()
    if report.empty:
        return report.reset_index()

    reference_rate = float(report["approval_rate"].max())
    report["adverse_impact_ratio"] = (
        report["approval_rate"] / reference_rate if reference_rate else np.nan
    )
    report["below_four_fifths"] = report["adverse_impact_ratio"] < 0.80
    return report.reset_index().sort_values(
        ["adverse_impact_ratio", "n"], ascending=[True, False]
    )


def audit_summary(reports: dict[str, pd.DataFrame]) -> dict:
    """Create a compact JSON-serialisable summary of proxy audit tables."""
    proxies = {}
    for name, report in reports.items():
        if "below_four_fifths" in report:
            flagged = report[report["below_four_fifths"].astype(bool)]
        else:
            flagged = report.iloc[0:0]
        minimum = (
            float(report["adverse_impact_ratio"].min()) if len(report) else None
        )
        proxies[name] = {
            "groups_analyzed": int(len(report)),
            "minimum_adverse_impact_ratio": minimum,
            "groups_below_four_fifths": (
                flagged["group"].astype(str).tolist() if "group" in flagged else []
            ),
        }
    return {
        "method": "approval rate relative to the highest-approval group",
        "screening_threshold": 0.80,
        "proxies": proxies,
        "caveat": (
            "State and masked ZIP are geographic proxies, not protected-class "
            "labels. This screen can reveal geographic disparities but cannot "
            "establish demographic parity or regulatory compliance."
        ),
    }
