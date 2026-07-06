"""
Post-hoc Youden threshold analysis on finetune_results.json.

Computes per-fold and aggregate metrics at the Youden-index optimal threshold
from the saved y_true/y_prob arrays. Outputs:
  - results/tables/youden_metrics.csv
  - Prints summary to stdout

Usage: python -m scripts.compute_youden
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (roc_curve, roc_auc_score, f1_score,
                             accuracy_score, matthews_corrcoef,
                             confusion_matrix, average_precision_score,
                             balanced_accuracy_score)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
TABLES_DIR   = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def youden_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute metrics at Youden-index optimal threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.argmax(tpr - fpr)
    opt = float(thresholds[idx])
    y_pred = (y_prob >= opt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold":    opt,
        "sensitivity":  tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity":  tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "accuracy":     accuracy_score(y_true, y_pred),
        "bal_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1":           f1_score(y_true, y_pred, average="macro"),
        "mcc":          matthews_corrcoef(y_true, y_pred),
        "auroc":        roc_auc_score(y_true, y_prob),
        "auprc":        average_precision_score(y_true, y_prob),
    }


def main():
    ft_path = RESULTS_DIR / "finetune_results.json"
    if not ft_path.exists():
        print(f"[youden] {ft_path} not found — run finetune first.")
        return

    data = json.load(open(ft_path))
    per_fold = data["per_fold"]
    print(f"[youden] Loaded {len(per_fold)} fold results from {ft_path}")

    rows = []
    y_true_all, y_prob_all = [], []
    for r in per_fold:
        y_true = np.array(r["y_true"])
        y_prob = np.array(r["y_prob"])
        y_true_all.append(y_true)
        y_prob_all.append(y_prob)

        ym = youden_metrics(y_true, y_prob)
        row = {"fold": r["fold"]}
        row.update(ym)
        # also include the default-threshold metrics for comparison
        row["default_sensitivity"] = r["metrics"].get("sensitivity", None)
        row["default_specificity"] = r["metrics"].get("specificity", None)
        row["default_accuracy"]    = r["metrics"].get("accuracy", None)
        row["default_f1"]          = r["metrics"].get("f1", None)
        rows.append(row)

    df = pd.DataFrame(rows)

    print("\n=== Per-fold Youden metrics ===")
    print(df[["fold", "threshold", "sensitivity", "specificity",
              "accuracy", "f1", "mcc", "auroc"]].to_string(index=False, float_format="{:.4f}".format))

    print("\n=== Mean ± Std (Youden) ===")
    metrics_cols = ["sensitivity", "specificity", "accuracy", "bal_accuracy", "f1", "mcc", "auroc", "auprc"]
    for m in metrics_cols:
        vals = df[m].values
        print(f"  {m:<20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Aggregate — apply Youden threshold from pooled distribution
    y_true_all = np.concatenate(y_true_all)
    y_prob_all = np.concatenate(y_prob_all)
    print("\n=== Aggregate (pooled across all folds, single Youden threshold) ===")
    agg = youden_metrics(y_true_all, y_prob_all)
    for k, v in agg.items():
        print(f"  {k:<20s}: {v:.4f}" if isinstance(v, float) else f"  {k:<20s}: {v}")

    # Save CSV
    csv_path = TABLES_DIR / "youden_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[youden] saved -> {csv_path}")


if __name__ == "__main__":
    main()
