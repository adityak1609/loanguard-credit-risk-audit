"""Audit approval outcomes across state and masked ZIP geography.

The audited fields are monitoring-only and are excluded from every production
feature set. Run after prepare_data.py and train_calibrate.py.

Run: python scripts/fairness_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd

from loanguard import config as C
from loanguard.fairness import audit_summary, group_report
from loanguard.features import FeatureSpec
from loanguard.model import split


def main() -> None:
    df = pd.read_parquet(C.PROCESSED / "loans.parquet")
    missing = sorted(set(C.FAIRNESS_PROXIES) - set(df.columns))
    if missing:
        raise RuntimeError(
            "Processed data predates the fairness audit. Run "
            f"`python scripts/prepare_data.py` to add: {', '.join(missing)}"
        )

    _, _, test = split(df)
    model = joblib.load(C.PROCESSED / "lgbm_model.pkl")
    calibrator = joblib.load(C.PROCESSED / "calibrator.pkl")
    spec = FeatureSpec.load(C.PROCESSED / "feature_spec.json")
    serving = json.loads((C.PROCESSED / "serving_config.json").read_text())

    raw_score = model.predict_proba(spec.transform(test))[:, 1]
    probability = calibrator.predict(raw_score)
    approved = probability < float(serving["threshold"])

    reports = {
        "addr_state": group_report(
            test["addr_state"],
            approved,
            target=test["target"],
            score=probability,
            min_group_size=500,
        ),
        "zip_code": group_report(
            test["zip_code"],
            approved,
            target=test["target"],
            score=probability,
            min_group_size=500,
        ),
    }

    C.REPORTS.mkdir(parents=True, exist_ok=True)
    for proxy, report in reports.items():
        report.to_csv(C.REPORTS / f"fairness_{proxy}.csv", index=False)

    summary = audit_summary(reports)
    summary.update(
        {
            "n_test": int(len(test)),
            "overall_approval_rate": float(approved.mean()),
            "threshold": float(serving["threshold"]),
            "feature_exclusion_verified": all(
                proxy not in spec.numeric + spec.categorical + spec.columns
                for proxy in C.FAIRNESS_PROXIES
            ),
        }
    )
    (C.REPORTS / "fairness_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"test loans: {len(test):,} | approval rate: {approved.mean():.2%}")
    for proxy, report in reports.items():
        flagged = int(report["below_four_fifths"].sum())
        minimum = report["adverse_impact_ratio"].min()
        print(
            f"{proxy}: {len(report)} groups | minimum ratio {minimum:.3f} | "
            f"four-fifths flags {flagged}"
        )
    print("Geographic proxies are excluded from the model; see MODEL_CARD.md for caveats.")


if __name__ == "__main__":
    main()
