"""
Quick monitor: parse finetune_full.log and show current training status.
Run any time to see: folds completed, current fold epoch, metrics.

Usage: python scripts/monitor_progress.py
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG = PROJECT_ROOT / "logs" / "finetune_full.log"
RESULTS = PROJECT_ROOT / "results" / "finetune_results.json"
CKPTS = PROJECT_ROOT / "checkpoints"


def main():
    # ── checkpoint count ──────────────────────────────────────────────────────
    fold_ckpts = sorted(CKPTS.glob("fold_*_best.pt"))
    print(f"Fold checkpoints saved: {len(fold_ckpts)}/10")
    for p in fold_ckpts:
        print(f"  {p.name}  ({p.stat().st_size // 1024 // 1024} MB)")

    # ── results JSON ─────────────────────────────────────────────────────────
    if RESULTS.exists():
        data = json.load(open(RESULTS))
        pf = data.get("per_fold", [])
        print(f"\nFolds in results JSON: {len(pf)}")
        for r in pf:
            m = r["metrics"]
            print(f"  Fold {r['fold']:2d}: acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
                  f"auroc={m['auroc']:.4f}  sens={m['sensitivity']:.4f}  "
                  f"spec={m['specificity']:.4f}")
        if len(pf) > 1:
            import numpy as np
            aucs = [r["metrics"]["auroc"] for r in pf]
            accs = [r["metrics"]["accuracy"] for r in pf]
            f1s  = [r["metrics"]["f1"] for r in pf]
            print(f"\n  MEAN so far: auroc={np.mean(aucs):.4f}±{np.std(aucs):.4f}  "
                  f"acc={np.mean(accs):.4f}  f1={np.mean(f1s):.4f}")

    # ── current fold progress ─────────────────────────────────────────────────
    if LOG.exists():
        with open(LOG, "r", errors="replace") as f:
            lines = f.readlines()

        # find current fold
        current_fold = None
        for l in reversed(lines):
            m = re.search(r"FOLD (\d+)/", l)
            if m:
                current_fold = int(m.group(1))
                break

        # last few epoch lines for current fold
        if current_fold:
            prefix = f"fold {current_fold} ep"
            ep_lines = [l.rstrip() for l in lines if prefix in l]
            print(f"\nCurrent fold {current_fold} — last 5 epochs:")
            for l in ep_lines[-5:]:
                print(f"  {l.strip()}")

        # test results already logged
        test_lines = [l.rstrip() for l in lines if "FOLD" in l and "TEST" in l]
        if test_lines:
            print(f"\nCompleted fold TEST results:")
            for l in test_lines:
                print(f"  {l.strip()}")


if __name__ == "__main__":
    main()
