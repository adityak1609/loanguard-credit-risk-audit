"""Build the modelling table from the raw LendingClub CSV.

Writes processed/loans.parquet plus a data summary to reports/.
Run: python scripts/prepare_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loanguard import config as C
from loanguard import data as D
from loanguard.features import engineer


def main() -> None:
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    C.REPORTS.mkdir(parents=True, exist_ok=True)

    print(f"reading {C.RAW_CSV.name} ...")
    raw = D.load_raw()
    print(f"  raw rows: {len(raw):,}")

    df = D.build(raw)
    print(f"  resolved loans: {len(df):,}")

    lgd = D.estimate_lgd(df)
    print(f"  empirical LGD: {lgd:.3f}")

    df = D.add_costs(df, lgd)
    df = engineer(df)

    out = C.PROCESSED / "loans.parquet"
    df.to_parquet(out, index=False)
    print(f"  wrote {out}")

    by_year = (
        df.groupby("issue_year")
        .agg(n=("target", "size"), default_rate=("target", "mean"),
             matured_share=("matured", "mean"))
        .round(4)
    )
    summary = {
        "n_rows": int(len(df)),
        "default_rate": float(df["target"].mean()),
        "imbalance_ratio": float((1 - df["target"].mean()) / df["target"].mean()),
        "empirical_lgd": lgd,
        "missing_share": {
            c: float(df[c].isna().mean())
            for c in C.APPLICANT_NUMERIC + C.INCUMBENT
            if c in df.columns and df[c].isna().any()
        },
        "by_year": json.loads(by_year.reset_index().to_json(orient="records")),
    }
    (C.REPORTS / "data_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nby issue year:")
    print(by_year.to_string())
    print(f"\ndefault rate: {summary['default_rate']:.2%} "
          f"({summary['imbalance_ratio']:.2f}x imbalance)")

    # ── fixed observation window ─────────────────────────────────────────
    print(f"\nbuilding {C.HORIZON_MONTHS}-month observation window target ...")
    hz = D.build_horizon(raw)
    hz = engineer(hz)
    hz.to_parquet(C.PROCESSED / "loans_12m.parquet", index=False)
    print(f"  eligible loans: {len(hz):,} (vs {len(df):,} resolved-only)")
    print(f"  default rate:   {hz['target'].mean():.2%}")

    sens = D.horizon_sensitivity(raw)
    sens.to_csv(C.REPORTS / "horizon_sensitivity.csv", index=False)
    print("\n  delinquency-lag sensitivity:")
    print(sens.to_string(index=False))

    hz_year = (
        hz.groupby("issue_year")
        .agg(n=("target", "size"), default_rate=("target", "mean"))
        .round(4)
    )
    hz_year.to_csv(C.REPORTS / "horizon_by_year.csv")
    print("\n  by issue year (every vintage observed for exactly 12 months):")
    print(hz_year.to_string())


if __name__ == "__main__":
    main()
