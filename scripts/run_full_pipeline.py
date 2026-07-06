"""
Run the full post-SSL evaluation pipeline:
1. Supervised 10-fold fine-tuning (uses SSL checkpoint)
2. Cross-dataset external validation (ensemble of fold checkpoints)
3. Ablation study (5 variants, 5 folds, 30 epochs)
4. Generate all plots

Usage:
    python -m scripts.run_full_pipeline [--ssl-ckpt checkpoints/ssl_pretrain.pt]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def run(cmd: list[str], log_path: Path, tag: str) -> int:
    print(f"\n{'='*70}")
    print(f"  [{tag}] STARTING")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"  LOG: {log_path}")
    print(f"{'='*70}")
    t0 = time.time()
    with open(log_path, "w") as logf:
        ret = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              cwd=str(PROJECT_ROOT))
    elapsed = (time.time() - t0) / 60
    status = "OK" if ret == 0 else f"FAILED (exit {ret})"
    print(f"  [{tag}] {status} — {elapsed:.1f} min")
    return ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl-ckpt", default="checkpoints/ssl_pretrain.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--abl-folds", type=int, default=5, help="Folds for ablation (faster)")
    ap.add_argument("--abl-epochs", type=int, default=40, help="Max epochs for ablation")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    python = sys.executable

    # ── Stage 1: Fine-tuning ──────────────────────────────────────────────────
    ret = run(
        [python, "-u", "-m", "scripts.finetune",
         "--ssl-ckpt", args.ssl_ckpt,
         "--epochs", str(args.epochs),
         "--batch", str(args.batch),
         "--n-folds", str(args.n_folds)],
        LOG_DIR / "finetune_full.log",
        "FINETUNE"
    )
    if ret != 0:
        print("FINETUNE failed — aborting pipeline.")
        sys.exit(1)

    # ── Stage 2: External validation ─────────────────────────────────────────
    if not args.skip_eval:
        run(
            [python, "-m", "scripts.evaluate", "--ensemble",
             "--max-samples", "50000",
             "--exclude-training-subjects",
             "--datasets", "paired_binary", "pcg_binary", "ecg_binary"],
            LOG_DIR / "evaluate_full.log",
            "EVALUATE"
        )

    # ── Stage 3: Ablation ─────────────────────────────────────────────────────
    if not args.skip_ablation:
        run(
            [python, "-m", "scripts.ablation",
             "--ssl-ckpt", args.ssl_ckpt,
             "--folds", str(args.abl_folds),
             "--epochs", str(args.abl_epochs)],
            LOG_DIR / "ablation_full.log",
            "ABLATION"
        )

    # ── Stage 4: Visualisations ───────────────────────────────────────────────
    run(
        [python, "-c",
         "import sys; sys.path.insert(0,'D:/AI_LAB_RP/CardioFusion-SSL'); "
         "from utils.visualise import generate_all; generate_all()"],
        LOG_DIR / "visualise_full.log",
        "VISUALISE"
    )

    print("\n" + "="*70)
    print("  FULL PIPELINE COMPLETE")
    print("  Check: results/finetune_results.json")
    print("         results/external_validation.json")
    print("         results/ablation_results.json")
    print("         results/plots/")
    print("="*70)


if __name__ == "__main__":
    main()
