"""Launch 10 v2 fine-tuning folds in parallel — one subprocess per fold.

Each subprocess trains one fold with N seeds inner-loop (default 1) via
finetune_v2, and writes results/fold_results_v2/fold_{N}.json when done.

Usage:
    # 10 folds, 1 seed each = 10 models (~1.5h on RTX 5070 Ti)
    python -m scripts.run_parallel_folds_v2

    # 10 folds, 3 seeds each = 30 models (~4h wall clock)
    python -m scripts.run_parallel_folds_v2 --seeds 3

    # Only re-run failed folds
    python -m scripts.run_parallel_folds_v2 --folds 3,7 --skip-done
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FOLD_LOG_DIR = PROJECT_ROOT / "logs" / "folds_v2"
FOLD_LOG_DIR.mkdir(parents=True, exist_ok=True)
FOLD_RESULT_DIR = PROJECT_ROOT / "results" / "fold_results_v2"
FOLD_RESULT_DIR.mkdir(parents=True, exist_ok=True)


def run_fold(fold_num_1idx: int, args) -> tuple[int, bool, str]:
    """Launch finetune_v2.py --fold-only for a single fold in a subprocess."""
    log_path = FOLD_LOG_DIR / f"fold_{fold_num_1idx}.log"

    # finetune_v2 uses 0-indexed --fold-only
    cmd = [
        sys.executable, "-m", "scripts.finetune_v2",
        "--fold-only",  str(fold_num_1idx - 1),
        "--folds",      str(args.n_folds),
        "--seeds",      str(args.seeds),
        "--epochs",     str(args.epochs),
        "--patience",   str(args.patience),
        "--batch",      str(args.batch),
        "--ssl-ckpt",   args.ssl_ckpt,
    ]

    t0 = time.time()
    print(f"[parallel_v2] fold {fold_num_1idx:2d} START  -> {log_path.name}", flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    with open(log_path, "w", buffering=1, encoding="utf-8") as flog:
        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT),
            stdout=flog, stderr=subprocess.STDOUT,
            env=env,
        )
    elapsed = time.time() - t0
    ok = proc.returncode == 0
    status = "DONE" if ok else f"FAILED (rc={proc.returncode})"
    print(f"[parallel_v2] fold {fold_num_1idx:2d} {status}  ({elapsed/60:.1f} min)",
          flush=True)
    return fold_num_1idx, ok, str(log_path)


def collect_results(n_folds: int) -> None:
    """Merge per-fold JSONs into a single finetune_v2_results.json."""
    import json
    import numpy as np

    all_records = []
    for n in range(1, n_folds + 1):
        p = FOLD_RESULT_DIR / f"fold_{n}.json"
        if not p.exists():
            print(f"[collect] fold {n}: no results file")
            continue
        d = json.load(open(p))
        all_records.extend(d.get("results", []))

    if not all_records:
        print("[collect] no results to aggregate")
        return

    # aggregate metrics across all (fold, seed) pairs
    agg = {}
    metrics_keys = list(all_records[0]["metrics"].keys())
    for m in metrics_keys:
        vals = [r["metrics"][m] for r in all_records if m in r["metrics"]]
        if vals:
            agg[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {
        "n_folds": n_folds,
        "n_models_total": len(all_records),
        "aggregated": agg,
        "per_fold_seed": all_records,
    }
    out_path = PROJECT_ROOT / "results" / "finetune_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[collect] {len(all_records)} models aggregated -> {out_path}")

    print(f"\n{'=' * 60}\n  SUMMARY across {len(all_records)} models\n{'=' * 60}")
    for m in ("auroc", "f1", "sensitivity", "specificity", "accuracy", "mcc"):
        if m in agg:
            print(f"  {m:<15s}: {agg[m]['mean']:.4f} ± {agg[m]['std']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all",
                    help="Comma-separated 1-indexed fold numbers, e.g. '3,4,5', or 'all'")
    ap.add_argument("--n-folds",      type=int, default=10)
    ap.add_argument("--seeds",        type=int, default=1,
                    help="Random seeds per fold (default 1 = Tier 2 only; 3 = full Tier 3)")
    ap.add_argument("--epochs",       type=int, default=100)
    ap.add_argument("--patience",     type=int, default=20)
    ap.add_argument("--batch",        type=int, default=32)
    ap.add_argument("--max-parallel", type=int, default=10)
    ap.add_argument("--ssl-ckpt",     default="checkpoints/ssl_pretrain_v2.pt")
    ap.add_argument("--skip-done",    action="store_true")
    ap.add_argument("--collect-only", action="store_true",
                    help="Skip training; just aggregate existing per-fold JSONs.")
    args = ap.parse_args()

    if args.collect_only:
        collect_results(args.n_folds)
        return

    # Determine which folds to run
    if args.folds == "all":
        folds_to_run = list(range(1, args.n_folds + 1))
    else:
        folds_to_run = [int(x.strip()) for x in args.folds.split(",")]

    if args.skip_done:
        before = len(folds_to_run)
        folds_to_run = [n for n in folds_to_run
                        if not (FOLD_RESULT_DIR / f"fold_{n}.json").exists()]
        skipped = before - len(folds_to_run)
        if skipped:
            print(f"[parallel_v2] skipping {skipped} already-completed folds")

    if not folds_to_run:
        print("[parallel_v2] Nothing to run — aggregating.")
        collect_results(args.n_folds)
        return

    n_workers = min(len(folds_to_run), args.max_parallel)
    print(f"[parallel_v2] Launching {len(folds_to_run)} folds × {args.seeds} seeds "
          f"= {len(folds_to_run) * args.seeds} models total")
    print(f"[parallel_v2] Concurrency: {n_workers} parallel folds")
    print(f"[parallel_v2] Logs -> {FOLD_LOG_DIR}")
    print(f"[parallel_v2] SSL checkpoint: {args.ssl_ckpt}")
    print()

    t0 = time.time()
    failed = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(run_fold, n, args): n for n in folds_to_run}
        for fut in as_completed(futures):
            fold_num, ok, _ = fut.result()
            if not ok:
                failed.append(fold_num)

    elapsed = time.time() - t0
    print(f"\n[parallel_v2] All folds finished in {elapsed/60:.1f} min")
    if failed:
        print(f"[parallel_v2] WARNING — {len(failed)} fold(s) FAILED: {failed}")
        print("[parallel_v2] Check logs in", FOLD_LOG_DIR)

    print("\n[parallel_v2] Aggregating ...")
    collect_results(args.n_folds)


if __name__ == "__main__":
    main()
