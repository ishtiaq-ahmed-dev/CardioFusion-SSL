"""Print all paper-ready numbers from completed training runs.

Reads finetune_results.json, external_validation_results.json, and
ablation_results.json and prints formatted numbers for every ←[FILL]
placeholder in the paper.

Usage:
    python -m scripts.report_results
    python -m scripts.report_results --youden   # use Youden threshold metrics
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def fmt(v, digits=4):
    if v is None:
        return "←[MISSING]"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def fmt_pct(v, digits=1):
    if v is None:
        return "←[MISSING]"
    return f"{v * 100:.{digits}f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--youden", action="store_true",
                    help="Report Youden-threshold metrics alongside default-threshold")
    args = ap.parse_args()

    ft_path = RESULTS_DIR / "finetune_results.json"
    ext_path = RESULTS_DIR / "external_validation_results.json"
    abl_path = RESULTS_DIR / "ablation_results.json"

    # ------------------------------------------------------------------ fine-tuning results
    print("\n" + "="*70)
    print("  FINE-TUNING RESULTS — for sections 5.2, Tables 2, S2, abstract")
    print("="*70)

    if not ft_path.exists():
        print(f"  [NOT FOUND] {ft_path}")
        print("  -> Run: python -m scripts.collect_fold_results")
    else:
        ft = json.load(open(ft_path))
        agg = ft["aggregated"]
        per_fold = ft["per_fold"]

        primary = ["accuracy", "bal_accuracy", "f1", "sensitivity", "specificity",
                   "auroc", "auprc", "mcc"]

        print(f"\n  N folds completed: {ft.get('n_folds_completed', len(per_fold))}")
        print("\n  AGGREGATED (mean ± std across 10 folds):")
        for m in primary:
            if m in agg:
                mean = agg[m]["mean"]
                std = agg[m]["std"]
                print(f"    {m:<20s}: {fmt(mean)} ± {fmt(std)}")

        # Bootstrap CI (from first fold's ci dict — global CI isn't stored in agg)
        # Actually bootstrap is per-fold; the aggregate CI requires the full arrays
        print("\n  PER-FOLD TEST METRICS (for Table S2):")
        header = f"  {'Fold':<6} {'Acc':>7} {'BalAcc':>8} {'F1':>7} {'Sens':>7} {'Spec':>7} {'AUROC':>7} {'AUPRC':>7} {'MCC':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in per_fold:
            m = r["metrics"]
            print(f"  {r['fold']:<6} "
                  f"{fmt(m.get('accuracy')):>7} "
                  f"{fmt(m.get('bal_accuracy')):>8} "
                  f"{fmt(m.get('f1')):>7} "
                  f"{fmt(m.get('sensitivity')):>7} "
                  f"{fmt(m.get('specificity')):>7} "
                  f"{fmt(m.get('auroc')):>7} "
                  f"{fmt(m.get('auprc')):>7} "
                  f"{fmt(m.get('mcc')):>7}")

        if args.youden:
            print("\n  PER-FOLD YOUDEN-THRESHOLD METRICS:")
            for r in per_fold:
                m = r["metrics"]
                yt = m.get("youden_threshold", "?")
                ys = m.get("youden_sensitivity", "?")
                ysp = m.get("youden_specificity", "?")
                yf1 = m.get("youden_f1", "?")
                print(f"    Fold {r['fold']}: threshold={fmt(yt, 3)}  "
                      f"sens={fmt(ys)}  spec={fmt(ysp)}  f1={fmt(yf1)}")

        # Training time from logs
        print("\n  TRAINING TIME (per fold):")
        fold_log_dir = PROJECT_ROOT / "logs" / "folds"
        for fold_idx in range(1, 11):
            log_path = fold_log_dir / f"fold_{fold_idx}.log"
            if log_path.exists():
                lines = open(log_path).readlines()
                last = [l for l in lines if "ep " in l]
                if last:
                    last_line = last[-1].strip()
                    n_epochs = int(last_line.split("ep ")[1].split()[0])
                    print(f"    Fold {fold_idx}: {n_epochs} epochs (→ estimate {n_epochs * 1.0:.0f} min)")

    # ------------------------------------------------------------------ external validation
    print("\n" + "="*70)
    print("  EXTERNAL VALIDATION — for section 5.4, Table 4, section 6.4")
    print("="*70)

    if not ext_path.exists():
        print(f"  [NOT FOUND] {ext_path}")
        print("  -> Run: python -m scripts.run_external_eval")
    else:
        ext = json.load(open(ext_path))
        dataset_names = {
            "circor":           "CirCor DigiScope (PCG-only, paediatric)",
            "cinc2016_external": "CinC2016 training-b/c/d/e/f (PCG-only)",
            "chapman":          "Chapman-Shaoxing (ECG-only)",
            "ptbxl":            "PTB-XL (ECG-only)",
            "mitbih":           "MIT-BIH Arrhythmia (ECG-only)",
            "cpsc2018":         "CPSC-2018 (ECG-only)",
        }
        print(f"\n  {'Dataset':<45} {'N':>8}  {'AUROC':>7}  {'F1':>7}  {'Sens':>7}  {'Spec':>7}")
        print("  " + "-"*90)
        for key, label in dataset_names.items():
            res = ext.get(key, {})
            if res.get("status") == "ok":
                m = res["overall"]
                n = res["n_samples"]
                print(f"  {label:<45} {n:>8,}  "
                      f"{fmt(m.get('auroc')):>7}  "
                      f"{fmt(m.get('f1')):>7}  "
                      f"{fmt(m.get('sensitivity')):>7}  "
                      f"{fmt(m.get('specificity')):>7}")
            else:
                print(f"  {label:<45} SKIPPED ({res.get('status', 'not_found')})")

    # ------------------------------------------------------------------ ablation
    print("\n" + "="*70)
    print("  ABLATION STUDY — for section 5.5, Table 5")
    print("="*70)

    if not abl_path.exists():
        print(f"  [NOT FOUND] {abl_path}")
        print("  -> Run: python -m scripts.ablation")
    else:
        abl = json.load(open(abl_path))
        full = abl.get("full_model", {})
        full_auroc = full.get("auroc", None)
        full_f1 = full.get("f1", None)

        variant_order = [
            ("full_model",    "Full model (CardioFusion-SSL)"),
            ("no_ssl",        "No SSL pretraining (random init)"),
            ("single_scale",  "Single-scale fusion (s=16 only)"),
            ("early_fusion",  "Early fusion (concat, no cross-attn)"),
            ("ecg_only",      "ECG only (PCG missing-token)"),
            ("pcg_only",      "PCG only (ECG missing-token)"),
        ]

        print(f"\n  {'Variant':<40} {'AUROC':>7}  {'ΔAUROC':>8}  {'F1':>7}")
        print("  " + "-"*70)
        for key, label in variant_order:
            v = abl.get(key, {})
            auroc = v.get("auroc", None)
            f1 = v.get("f1", None)
            delta = f"{(auroc - full_auroc):+.4f}" if (auroc is not None and full_auroc is not None and key != "full_model") else "—"
            print(f"  {label:<40} {fmt(auroc):>7}  {delta:>8}  {fmt(f1):>7}")

    print("\n" + "="*70)
    print("  DONE — copy numbers above into paper ←[FILL] placeholders")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
