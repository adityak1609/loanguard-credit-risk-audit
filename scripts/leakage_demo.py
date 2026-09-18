"""Why published AUCs on this dataset are not comparable to ours.

Public notebooks on the LendingClub extract routinely report 0.95+ AUC. That
number is reachable in about ten lines, and it is meaningless. The dataset
ships with columns recorded *after* the loan performed:

  last_fico_range_low/high  borrower's credit score at the most recent pull.
                            For a defaulted loan that pull happens after the
                            default, so the feature partly *is* the label.
  last_pymnt_amnt           size of the final payment received.
  out_prncp                 principal still outstanding -- zero for every
                            resolved loan, which encodes resolution itself.
  total_pymnt, total_rec_int  cumulative amounts actually collected.

None exist when the lending decision is made. A model using them cannot be
deployed; it can only score loans whose outcome is already known.

This script trains the legitimate model and the leaky one on the same split
and prints both, so the gap is a measured artefact rather than an assertion.

Run: python scripts/leakage_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from loanguard import config as C
from loanguard import evaluate as E
from loanguard.features import FeatureSpec
from loanguard.model import fit_lgbm, split


def main() -> None:
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(C.PROCESSED / "loans.parquet")
    train, val, test = split(df)

    rows = []
    models = {}
    for name in ["full", "full_leaky"]:
        spec = FeatureSpec.build(name, train)
        model = fit_lgbm(spec, train, val)
        models[name] = (model, spec)
        p = model.predict_proba(spec.transform(test))[:, 1]
        row = {"feature_set": name, "valid": name == "full"}
        row.update(E.ranking_metrics(test["target"].values, p))
        row.update(E.calibration_metrics(test["target"].values, p))
        rows.append(row)
        print(f"  {name:12s} AUC {row['auc']:.4f}  AP {row['average_precision']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(C.REPORTS / "leakage_demo.csv", index=False)

    # Which leaked columns are doing the work?
    model, spec = models["full_leaky"]
    imp = (
        pd.Series(model.feature_importances_, index=spec.columns)
        .sort_values(ascending=False)
        .head(10)
    )
    leaky_in_top = [c for c in imp.index if c in C.LEAKY]

    valid_auc = float(out.loc[out.feature_set == "full", "auc"].iloc[0])
    leaky_auc = float(out.loc[out.feature_set == "full_leaky", "auc"].iloc[0])
    (C.REPORTS / "leakage_demo.json").write_text(json.dumps({
        "valid_auc": valid_auc,
        "leaky_auc": leaky_auc,
        "inflation": leaky_auc - valid_auc,
        "leaky_columns": C.LEAKY,
        "leaky_columns_in_top10_importance": leaky_in_top,
        "verdict": ("The leaky model is not deployable: every added column is "
                    "recorded after origination. It is reported here only to "
                    "explain why published AUCs on this dataset are not "
                    "comparable to the 0.72 obtained legitimately."),
    }, indent=2), encoding="utf-8")

    print("\ntop-10 importance (leaky model), leaked columns marked:")
    for name, v in imp.items():
        mark = "  <-- LEAKED" if name in C.LEAKY else ""
        print(f"  {name:28s} {v:8.0f}{mark}")

    print(f"\nvalid model:  AUC {valid_auc:.4f}")
    print(f"leaky model:  AUC {leaky_auc:.4f}  (+{leaky_auc - valid_auc:.4f})")
    print("\nThe leaky model is NOT deployable. See module docstring.")


if __name__ == "__main__":
    main()
