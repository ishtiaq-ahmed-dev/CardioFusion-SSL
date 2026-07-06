"""Orchestrates the full Tier 2 + Tier 3 retrain.

Runs, in order:
  1. SSL pretraining v2 (200 epochs, masked-recon enabled)
  2. 10-fold × 3-seed supervised fine-tuning v2 (SpecAugment, MixUp, mod-dropout, SWA)
  3. External validation with 30-model ensemble
  4. Youden threshold analysis on new results

Expected wall-clock on RTX 5070 Ti (16 GB VRAM):
  Stage 1  (SSL 200 epochs)             ~2 hours
  Stage 2  (10 folds × 3 seeds)          ~4-6 hours parallel
  Stage 3  (external eval, 30 models)    ~1 hour
  Stage 4  (post-hoc Youden)             ~10 seconds
  TOTAL:                                 ~7-9 hours

Progress is logged to results/tier23_pipeline.log.
"""
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "results" / "tier23_pipeline.log"
LOG_PATH.parent.mkdir(exist_ok=True)


def run(cmd: list, stage: str, log_file):
    """Run a subprocess and stream output to both stdout and the log file."""
    header = f"\n{'=' * 70}\n  STAGE: {stage}\n  CMD:   {' '.join(cmd)}\n  START: {datetime.now().isoformat()}\n{'=' * 70}\n"
    print(header, flush=True)
    log_file.write(header)
    log_file.flush()

    # Force UTF-8 in the child process and tolerate any stray bytes
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        encoding="utf-8", errors="replace",
        env=env,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
        log_file.flush()
    proc.wait()

    footer = f"\n  END:   {datetime.now().isoformat()}   (exit {proc.returncode})\n"
    print(footer, flush=True)
    log_file.write(footer)
    log_file.flush()

    if proc.returncode != 0:
        raise SystemExit(f"Stage '{stage}' failed with exit code {proc.returncode}")


def main():
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        # ── Stage 1: SSL v2 ─────────────────────────────────────────
        run(
            [sys.executable, "-m", "scripts.pretrain_v2", "--epochs", "200"],
            "SSL pretraining v2 (200 epochs, masked-recon enabled)",
            log_file,
        )

        # ── Stage 2: fine-tune v2, 10 folds × 3 seeds ───────────────
        run(
            [sys.executable, "-m", "scripts.finetune_v2",
             "--folds", "10", "--seeds", "3", "--epochs", "100",
             "--patience", "20"],
            "Fine-tune v2 (10 folds × 3 seeds, SWA + augmentations)",
            log_file,
        )

        # ── Stage 3: post-hoc Youden ────────────────────────────────
        run(
            [sys.executable, "-m", "scripts.compute_youden"],
            "Youden threshold analysis on v2 results",
            log_file,
        )

        print("\n\n🎉 Tier 2/3 pipeline complete.", flush=True)


if __name__ == "__main__":
    main()
