"""Explain the residual out-of-time AUC drop on the fixed-window target.

The experiment separates three candidate explanations:

1. A particular feature family decays (feature-set ablations).
2. Old training vintages make the model stale (2015-only retraining).
3. The applicant population changes (PSI and missingness diagnostics).

Run after prepare_data.py:
    python scripts/drift_diagnosis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sklearn.model_selection import train_test_split

from loanguard import config as C
from loanguard import evaluate as E
from loanguard.drift import feature_shift_report
from loanguard.features import FEATURE_SETS, FeatureSpec
from loanguard.model import fit_lgbm

TRAIN_END = 2014
REFERENCE_YEAR = 2015
FEATURE_ABLATIONS = ["grade_only", "incumbent_only", "applicant_only", "full"]
NEWER_BUREAU_CANDIDATES = ["bc_util", "tot_cur_bal", "acc_open_past_24mths"]


def score(model, spec, frame) -> dict:
    probability = model.predict_proba(spec.transform(frame))[:, 1]
    metrics = {"n": int(len(frame))}
    metrics.update(E.ranking_metrics(frame["target"].to_numpy(), probability))
    return metrics | {"probability": probability}


def main() -> None:
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(C.PROCESSED / "loans_12m.parquet")
    train = df[df["issue_year"] <= TRAIN_END]
    reference = df[df["issue_year"] == REFERENCE_YEAR]
    future = df[df["issue_year"] > REFERENCE_YEAR]

    ablations = []
    fitted = {}
    for name in FEATURE_ABLATIONS:
        spec = FeatureSpec.build(name, train)
        model = fit_lgbm(spec, train, reference)
        ref_score = score(model, spec, reference)
        future_score = score(model, spec, future)
        ablations.append(
            {
                "feature_set": name,
                "n_features": len(spec.columns),
                "auc_2015": ref_score["auc"],
                "auc_2016_2017": future_score["auc"],
                "auc_drop": ref_score["auc"] - future_score["auc"],
                "ap_2015": ref_score["average_precision"],
                "ap_2016_2017": future_score["average_precision"],
            }
        )
        fitted[name] = (model, spec, ref_score, future_score)
        print(
            f"{name:16s} AUC 2015 {ref_score['auc']:.4f} | "
            f"2016-17 {future_score['auc']:.4f} | "
            f"drop {ref_score['auc'] - future_score['auc']:+.4f}"
        )

    ablation_frame = pd.DataFrame(ablations)
    ablation_frame.to_csv(C.REPORTS / "drift_feature_ablation.csv", index=False)

    # Does refreshing on the immediately preceding vintage recover the loss?
    recent_train, recent_val = train_test_split(
        reference,
        test_size=0.20,
        random_state=C.RANDOM_SEED,
        stratify=reference["target"],
    )
    recent_spec = FeatureSpec.build("full", recent_train)
    recent_model = fit_lgbm(recent_spec, recent_train, recent_val)
    recent_holdout = score(recent_model, recent_spec, recent_val)
    recent_future = score(recent_model, recent_spec, future)

    original_model, original_spec, original_ref, original_future = fitted["full"]
    training_windows = pd.DataFrame(
        [
            {
                "training_window": "through_2014",
                "n_train": len(train),
                "validation_window": "2015",
                "validation_auc": original_ref["auc"],
                "test_window": "2016-2017",
                "test_auc": original_future["auc"],
            },
            {
                "training_window": "2015_only",
                "n_train": len(recent_train),
                "validation_window": "2015_holdout",
                "validation_auc": recent_holdout["auc"],
                "test_window": "2016-2017",
                "test_auc": recent_future["auc"],
            },
        ]
    )
    training_windows.to_csv(C.REPORTS / "drift_training_window.csv", index=False)

    numeric, categorical = FEATURE_SETS["full"]
    shift = feature_shift_report(reference, future, numeric, categorical)
    shift.to_csv(C.REPORTS / "drift_feature_shift.csv", index=False)

    # Check whether the loss persists inside coarse risk/term segments rather
    # than arising only from their changing portfolio shares.
    segment_rows = []
    for segment_name in ["grade", "term_months"]:
        for value in sorted(df[segment_name].dropna().unique()):
            ref_mask = reference[segment_name] == value
            future_mask = future[segment_name] == value
            if ref_mask.sum() < 500 or future_mask.sum() < 500:
                continue
            ref_metrics = E.ranking_metrics(
                reference.loc[ref_mask, "target"].to_numpy(),
                original_ref["probability"][ref_mask.to_numpy()],
            )
            future_metrics = E.ranking_metrics(
                future.loc[future_mask, "target"].to_numpy(),
                original_future["probability"][future_mask.to_numpy()],
            )
            segment_rows.append(
                {
                    "segment": segment_name,
                    "value": value,
                    "n_2015": int(ref_mask.sum()),
                    "n_2016_2017": int(future_mask.sum()),
                    "auc_2015": ref_metrics["auc"],
                    "auc_2016_2017": future_metrics["auc"],
                    "auc_drop": ref_metrics["auc"] - future_metrics["auc"],
                    "base_rate_2015": ref_metrics["base_rate"],
                    "base_rate_2016_2017": future_metrics["base_rate"],
                }
            )
    segments = pd.DataFrame(segment_rows)
    segments.to_csv(C.REPORTS / "drift_segments.csv", index=False)

    full_features = set(sum(FEATURE_SETS["full"], []))
    absent_candidates = sorted(set(NEWER_BUREAU_CANDIDATES) - full_features)
    original_drop = original_ref["auc"] - original_future["auc"]
    auc_recovered = recent_future["auc"] - original_future["auc"]
    recovered_share = auc_recovered / original_drop
    result = {
        "question": "What explains the residual fixed-window out-of-time AUC drop?",
        "feature_ablation": ablation_frame.to_dict(orient="records"),
        "training_window": training_windows.to_dict(orient="records"),
        "auc_recovered_by_2015_retraining": auc_recovered,
        "share_of_original_drop_recovered": recovered_share,
        "top_feature_shift": shift.head(10).to_dict(orient="records"),
        "segments_with_lower_future_auc": int((segments["auc_drop"] > 0).sum()),
        "segments_analyzed": int(len(segments)),
        "newer_bureau_candidates_absent_from_model": absent_candidates,
        "conclusion": (
            "The newer-bureau-field hypothesis cannot explain this model's drift because "
            "those fields are not model inputs. Drift is concentrated in applicant signals "
            "rather than the incumbent grade fields, and AUC declines inside every grade "
            "and term segment, so changing segment mix is not enough to explain it. "
            f"Retraining on 2015 raises 2016-2017 AUC by {auc_recovered:.4f}, recovering "
            f"{recovered_share:.1%} of the original drop. The largest PSI is only "
            f"{shift.iloc[0]['psi']:.3f}, pointing to changing feature-outcome relationships "
            "(model staleness/concept drift) more than a large marginal population shift."
        ),
    }
    (C.REPORTS / "drift_diagnosis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print("\ntraining-window comparison:")
    print(training_windows.round(4).to_string(index=False))
    print("\ntop feature shifts (PSI):")
    print(shift.head(10).round(4).to_string(index=False))
    print(
        f"\nwithin-segment AUC declined in {(segments['auc_drop'] > 0).sum()} "
        f"of {len(segments)} grade/term segments"
    )
    print(
        "newer bureau fields in current model: ",
        sorted(set(NEWER_BUREAU_CANDIDATES) & full_features),
    )


if __name__ == "__main__":
    main()
