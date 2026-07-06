"""Run multiple fine-tuning folds in parallel — one subprocess per fold.

Each subprocess trains exactly one fold, saves its checkpoint and a per-fold
JSON to results/fold_results/fold_N.json, and exits. This script launches all
requested folds concurrently, waits for them all, then calls collect_fold_results
to merge the per-fold JSONs into results/finetune_results.json.

With 16 GB VRAM and ~900 MB per fold, up to 10 folds can run simultaneously
without OOM. Typical wall-clock time: ~3 hours for all 10 folds vs. ~30 hours
sequential.

Usage:
    # Run all 10 folds in parallel (default)
    python -m scripts.run_parallel_folds

    # Run only folds 3-10 (folds 1-2 already done)
    python -m scripts.run_parallel_folds --folds 3,4,5,6,7,8,9,10

    # Limit concurrency to avoid VRAM pressure
    python -m scripts.run_parallel_folds --max-parallel 5

    # Skip folds that already have a result JSON
    python -m scripts.run_parallel_folds --skip-done
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FOLD_LOG_DIR = PROJECT_ROOT / "logs" / "folds"
FOLD_LOG_DIR.mkdir(parents=True, exist_ok=True)

FOLD_RESULT_DIR = PROJECT_ROOT / "results" / "fold_results"
FOLD_RESULT_DIR.mkdir(parents=True, exist_ok=True)


def run_fold(fold_num: int, args) -> tuple[int, bool, str]:
    """Launch finetune.py for a single fold in a subprocess.

    Returns (fold_num, success, log_path).
    """
    log_path = FOLD_LOG_DIR / f"fold_{fold_num}.log"

    cmd = [
        sys.executable, "-m", "scripts.finetune",
        "--fold-indices", str(fold_num),
        "--n-folds",      str(args.n_folds),
        "--batch",        str(args.batch),
        "--epochs",       str(args.epochs),
        "--dl-workers",   str(args.dl_workers),
    ]
    if args.compile:
        cmd.append("--compile")
    if args.from_scratch:
        cmd.append("--from-scratch")

    t0 = time.time()
    print(f"[parallel] fold {fold_num:2d} START  -> {log_path.name}", flush=True)
    with open(log_path, "w", buffering=1) as flog:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=flog,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    ok = proc.returncode == 0
    status = "DONE" if ok else f"FAILED (rc={proc.returncode})"
    print(f"[parallel] fold {fold_num:2d} {status}  ({elapsed/60:.1f} min)", flush=True)
    return fold_num, ok, str(log_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CardioFusion-SSL fine-tuning folds in parallel.")
    ap.add_argument("--folds", default="all",
                    help="Comma-separated 1-indexed fold numbers, e.g. '3,4,5,6,7,8,9,10', or 'all'")
    ap.add_argument("--n-folds",      type=int, default=10)
    ap.add_argument("--batch",        type=int, default=32)
    ap.add_argument("--epochs",       type=int, default=80)
    ap.add_argument("--dl-workers",   type=int, default=0,
                    help="DataLoader num_workers per fold (0 = synchronous, safest on Windows)")
    ap.add_argument("--max-parallel", type=int, default=10,
                    help="Maximum concurrent fold processes (default 10, one per fold)")
    ap.add_argument("--compile",      action="store_true",
                    help="Enable torch.compile(mode='reduce-overhead') in each fold")
    ap.add_argument("--from-scratch", action="store_true",
                    help="Train without SSL pretrained weights")
    ap.add_argument("--skip-done",    action="store_true",
                    help="Skip folds that already have a result JSON in results/fold_results/")
    args = ap.parse_args()

    # Determine which folds to run
    if args.folds == "all":
        folds_to_run = list(range(1, args.n_folds + 1))
    else:
        folds_to_run = [int(x.strip()) for x in args.folds.split(",")]

    if args.skip_done:
        before = len(folds_to_run)
        folds_to_run = [
            n for n in folds_to_run
            if not (FOLD_RESULT_DIR / f"fold_{n}.json").exists()
        ]
        skipped = before - len(folds_to_run)
        if skipped:
            print(f"[parallel] Skipping {skipped} already-completed folds (--skip-done)")

    if not folds_to_run:
        print("[parallel] Nothing to run — all folds already complete.")
        subprocess.run([sys.executable, "-m", "scripts.collect_fold_results"],
                       cwd=str(PROJECT_ROOT))
        return

    n_workers = min(len(folds_to_run), args.max_parallel)
    print(f"[parallel] Launching {len(folds_to_run)} folds with {n_workers} workers")
    print(f"[parallel] Folds: {folds_to_run}")
    print(f"[parallel] batch={args.batch}  epochs={args.epochs}  "
          f"dl_workers={args.dl_workers}  compile={args.compile}")
    print(f"[parallel] Logs -> {FOLD_LOG_DIR}")
    print()

    t_start = time.time()
    failed = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(run_fold, n, args): n for n in folds_to_run}
        for fut in as_completed(futures):
            fold_num, ok, _ = fut.result()
            if not ok:
                failed.append(fold_num)

    elapsed_total = time.time() - t_start
    print(f"\n[parallel] All folds finished in {elapsed_total/60:.1f} min")
    if failed:
        print(f"[parallel] WARNING — {len(failed)} fold(s) FAILED: {failed}")
        print("[parallel] Check logs in", FOLD_LOG_DIR)

    # Aggregate all per-fold JSONs into finetune_results.json
    print("\n[parallel] Collecting results ...")
    subprocess.run(
        [sys.executable, "-m", "scripts.collect_fold_results"],
        cwd=str(PROJECT_ROOT),
        check=False,
    )


if __name__ == "__main__":
    main()
