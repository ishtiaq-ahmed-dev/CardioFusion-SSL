"""Aggregate per-fold result JSONs into finetune_results.json.

Reads results/fold_results/fold_N.json for all available folds,
merges them into the canonical results/finetune_results.json, and
prints a summary table.

Usage:
    python -m scripts.collect_fold_results
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

FOLD_RESULT_DIR = PROJECT_ROOT / "results" / "fold_results"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    result_files = sorted(
        FOLD_RESULT_DIR.glob("fold_*.json"),
        key=lambda p: int(p.stem.split("_")[1]),
    )

    if not result_files:
        print(f"[collect] No per-fold JSONs found in {FOLD_RESULT_DIR}")
        return

    print(f"[collect] Found {len(result_files)} fold result files:")
    for f in result_files:
        print(f"  {f.name}")

    all_results = []
    for p in result_files:
        with open(p) as fh:
            all_results.append(json.load(fh))

    # Aggregate numeric metrics
    metric_names = [
        k for k, v in all_results[0]["metrics"].items()
        if isinstance(v, (int, float))
    ]
    agg: dict = {}
    for m in metric_names:
        vals = []
        for r in all_results:
            v = r["metrics"].get(m)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            agg[m] = {
                "mean":     float(np.mean(vals)),
                "std":      float(np.std(vals)),
                "per_fold": vals,
            }

    out = {
        "aggregated":        agg,
        "per_fold":          all_results,
        "n_folds_completed": len(all_results),
    }

    out_path = RESULTS_DIR / "finetune_results.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[collect] Saved -> {out_path}")

    # Summary table
    primary = ["accuracy", "bal_accuracy", "f1", "sensitivity", "specificity",
               "auroc", "auprc", "mcc"]
    rows = []
    for r in all_results:
        row = {"fold": r["fold"]}
        for k in primary:
            v = r["metrics"].get(k)
            row[k] = f"{v:.4f}" if isinstance(v, float) else ""
        rows.append(row)

    # Mean ± std row
    row_agg = {"fold": "mean±std"}
    for k in primary:
        if k in agg:
            row_agg[k] = f"{agg[k]['mean']:.4f}±{agg[k]['std']:.4f}"
        else:
            row_agg[k] = ""
    rows.append(row_agg)

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    # Save CSV
    csv_path = TABLES_DIR / "kfold_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[collect] CSV -> {csv_path}")


if __name__ == "__main__":
    main()
