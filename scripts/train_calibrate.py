"""Train the production model, calibrate it, and pick a threshold by cost.

Three things the original pipeline got wrong, addressed here:

1. `scale_pos_weight` was set to the class ratio, which deliberately inflates
   predicted probabilities. The output was a ranking score, but the Streamlit
   app rendered it as "probability of default" -- so displayed risk was
   roughly 3x the true rate. We fit without the weight and calibrate instead.
2. Calibration was never measured. We report Brier and expected calibration
   error for the weighted model, the unweighted model, and the isotonic-
   calibrated model, on a held-out set the calibrator never saw.
3. The operating point was p = 0.5, chosen implicitly by `predict()`. We pick
   it by minimising expected loss under per-loan costs instead.

Run: python scripts/train_calibrate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import lightgbm as lgb
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression

from loanguard import config as C
from loanguard import evaluate as E
from loanguard.features import FeatureSpec
from loanguard.model import LGB_PARAMS, split


def main() -> None:
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(C.PROCESSED / "loans.parquet")
    train, val, test = split(df)

    spec = FeatureSpec.build("full", train)
    Xtr, Xva, Xte = (spec.transform(d) for d in (train, val, test))
    ytr, yva, yte = (d["target"].values for d in (train, val, test))

    # ── the two fits ─────────────────────────────────────────────────────
    ratio = (ytr == 0).sum() / (ytr == 1).sum()
    print(f"class ratio {ratio:.2f}x")

    fits = {}
    for label, extra in [
        ("weighted", {"scale_pos_weight": ratio}),
        ("unweighted", {}),
    ]:
        m = lgb.LGBMClassifier(**{**LGB_PARAMS, **extra})
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False)])
        fits[label] = m

    # ── calibrate the unweighted model on val, score on test ─────────────
    p_val = fits["unweighted"].predict_proba(Xva)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, yva)

    variants = {
        "weighted_raw": fits["weighted"].predict_proba(Xte)[:, 1],
        "unweighted_raw": fits["unweighted"].predict_proba(Xte)[:, 1],
        "unweighted_isotonic": iso.predict(fits["unweighted"].predict_proba(Xte)[:, 1]),
    }

    rows = []
    for name, p in variants.items():
        row = {"variant": name}
        row.update(E.ranking_metrics(yte, p))
        row.update(E.calibration_metrics(yte, p))
        rows.append(row)
    cal = pd.DataFrame(rows)
    cal.to_csv(C.REPORTS / "calibration.csv", index=False)
    print("\ncalibration (test set):")
    print(cal[["variant", "auc", "brier", "ece", "mean_predicted",
               "observed_rate"]].to_string(index=False))

    E.reliability_curve(yte, variants["weighted_raw"]).to_csv(
        C.REPORTS / "reliability_weighted.csv", index=False)
    E.reliability_curve(yte, variants["unweighted_isotonic"]).to_csv(
        C.REPORTS / "reliability_calibrated.csv", index=False)

    # ── threshold by expected loss, chosen on val, reported on test ──────
    p_val_cal = iso.predict(p_val)
    curve_val = E.cost_curve(yva, p_val_cal, val["cost_fn"], val["cost_fp"])
    chosen = E.select_threshold(curve_val)
    # Serve the midpoint of the near-optimal band rather than the raw argmin:
    # the band is wide and nearly flat, so the argmin is noise-driven and does
    # not reliably transfer from validation to test.
    lo, hi = chosen["flat_region"]
    t = round((lo + hi) / 2, 3)

    p_test_cal = variants["unweighted_isotonic"]
    curve_test = E.cost_curve(yte, p_test_cal, test["cost_fn"], test["cost_fp"])
    curve_test.to_csv(C.REPORTS / "cost_curve_test.csv", index=False)

    at_t = curve_test.iloc[(curve_test["threshold"] - t).abs().idxmin()]
    approve_all = float(curve_test["total_cost"].iloc[-1])
    at_half = curve_test.iloc[(curve_test["threshold"] - 0.5).abs().idxmin()]

    sens = E.lgd_sensitivity(
        yte, p_test_cal, test["loan_amnt"].values, test["cost_fp"].values
    )
    sens.to_csv(C.REPORTS / "lgd_sensitivity.csv", index=False)
    print("\nLGD sensitivity (test):")
    print(sens.round(4).to_string(index=False))

    decision = {
        "threshold_selected_on_val": t,
        "val": chosen,
        "test": {
            "total_cost": float(at_t["total_cost"]),
            "approval_rate": float(at_t["approval_rate"]),
            "cost_per_application": float(at_t["cost_per_application"]),
            "approve_all_cost": approve_all,
            "savings_vs_approve_all": approve_all - float(at_t["total_cost"]),
            "cost_at_p50": float(at_half["total_cost"]),
            "savings_vs_p50": float(at_half["total_cost"]) - float(at_t["total_cost"]),
            "confusion": E.confusion_at(yte, p_test_cal, t),
        },
    }
    (C.REPORTS / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8")

    print(f"\nthreshold {t:.3f} (selected on val, applied to test)")
    print(f"  approval rate      {at_t['approval_rate']:.1%}")
    print(f"  cost/application   ${at_t['cost_per_application']:,.0f}")
    print(f"  vs approve-all     ${decision['test']['savings_vs_approve_all']:,.0f} saved")
    print(f"  vs p=0.50 default  ${decision['test']['savings_vs_p50']:,.0f} saved")
    print(f"  book default rate  {decision['test']['confusion']['default_rate_in_book']:.2%}")

    # ── persist serving artefacts ────────────────────────────────────────
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    joblib.dump(fits["unweighted"], C.PROCESSED / "lgbm_model.pkl")
    joblib.dump(iso, C.PROCESSED / "calibrator.pkl")
    spec.save(C.PROCESSED / "feature_spec.json")
    (C.PROCESSED / "serving_config.json").write_text(
        json.dumps(
            {
                "threshold": t,
                "trained_on": "random split, all years",
                "artifact_versions": {
                    "lightgbm": lgb.__version__,
                    "scikit_learn": sklearn.__version__,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote model, calibrator, feature spec to {C.PROCESSED}")


if __name__ == "__main__":
    main()
