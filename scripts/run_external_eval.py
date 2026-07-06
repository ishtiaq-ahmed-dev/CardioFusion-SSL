"""Run systematic external validation for the CardioFusion-SSL paper.

Evaluates the 10-fold ensemble on each external dataset with proper
source filtering and no sampling cap. Produces per-dataset metrics
suitable for filling paper Table 4 and Table S2.

Datasets evaluated:
  PCG-only external:
    - CirCor DigiScope (circor, 63,478 windows)
    - CinC2016 training-b/c/d/e/f (cinc2016 excluding training-a subjects)
  ECG-only external:
    - Chapman-Shaoxing (chapman)
    - PTB-XL (ptbxl)
    - MIT-BIH Arrhythmia (mitbih)
    - CPSC-2018 (c2020_cpsc_2018 + c2020_cpsc_2018_extra)

Usage:
    python -m scripts.run_external_eval
    python -m scripts.run_external_eval --no-ensemble   # faster, per-fold only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from configs import CFG
from models.full_model import CardioFusionSSL
from data.adapter import load_cache, PairedCacheDataset, collate_with_strings
from scripts.finetune import compute_metrics, bootstrap_ci


def ensemble_evaluate_with_filter(
    ckpt_paths: list[str],
    cache_name: str,
    device: str,
    include_sources: list[str] | None = None,
    exclude_subjects: set | None = None,
    batch_size: int = 128,
    desc: str = "",
) -> dict:
    """Soft-voting ensemble evaluation with source/subject filtering."""
    print(f"\n  [{desc}] loading cache '{cache_name}' ...")
    try:
        arrays, meta = load_cache(cache_name)
    except FileNotFoundError:
        print(f"    SKIP: cache '{cache_name}' not found")
        return {"status": "not_found"}

    if "binary" not in meta.columns:
        print(f"    SKIP: no binary labels in {cache_name}")
        return {"status": "no_labels"}

    # build boolean mask
    mask = np.ones(len(meta), dtype=bool)

    if include_sources and "source" in meta.columns:
        mask &= meta["source"].isin(include_sources).values
        print(f"    source filter {include_sources}: {mask.sum()} / {len(meta)} rows")

    if exclude_subjects and "subject" in meta.columns:
        excl = meta["subject"].isin(exclude_subjects).values
        n_excl = excl[mask].sum()
        mask &= ~excl
        print(f"    excluded {n_excl} rows from {len(exclude_subjects)} training subjects")

    indices = np.where(mask)[0]
    if len(indices) == 0:
        print(f"    SKIP: no samples after filtering")
        return {"status": "empty_after_filter"}

    print(f"    N = {len(indices)} windows")

    ds = PairedCacheDataset(cache_name, indices=indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True, collate_fn=collate_with_strings)

    # collect per-model probabilities
    all_probs = []
    for ckpt_path in ckpt_paths:
        model = CardioFusionSSL(enable_ssl=False, enable_recon=False).to(device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(sd, strict=False)
        model.eval()

        fold_prob = []
        fold_true = []
        fold_src = []
        with torch.no_grad():
            for batch in loader:
                for k in ("ecg", "pcg_mel", "has_ecg", "has_pcg", "label"):
                    batch[k] = batch[k].to(device, non_blocking=True)
                with autocast("cuda", enabled=(device == "cuda" and CFG.AMP)):
                    out = model(batch, mode="supervised")
                prob = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
                fold_prob.append(prob)
                fold_true.append(batch["label"].cpu().numpy())
                fold_src.extend(batch.get("source", ["unknown"] * len(prob)))
        all_probs.append(np.concatenate(fold_prob))
        y_true = np.concatenate(fold_true)
        sources = np.array(fold_src)
        del model
        torch.cuda.empty_cache()

    # soft-vote
    avg_prob = np.mean(all_probs, axis=0)
    avg_pred = (avg_prob >= 0.5).astype(int)

    valid = y_true >= 0
    y_true = y_true[valid]
    y_pred = avg_pred[valid]
    y_prob = avg_prob[valid]
    sources = sources[valid]

    overall = compute_metrics(y_true, y_pred, y_prob)
    # Skip bootstrap CI for external eval — N can be 300K+ making it prohibitively slow.
    # CI is reported for the primary 10-fold CV test set only.
    ci = {}

    # per-source breakdown
    per_source = {}
    for src in np.unique(sources):
        m = sources == src
        if m.sum() >= 10:
            per_source[src] = compute_metrics(y_true[m], y_pred[m], y_prob[m])
            print(f"      {src}: N={m.sum()}  auroc={per_source[src]['auroc']:.4f}  "
                  f"f1={per_source[src]['f1']:.4f}  "
                  f"sens={per_source[src]['sensitivity']:.4f}  "
                  f"spec={per_source[src]['specificity']:.4f}")

    print(f"    OVERALL: N={len(y_true)}  auroc={overall['auroc']:.4f}  "
          f"f1={overall['f1']:.4f}  sens={overall['sensitivity']:.4f}  "
          f"spec={overall['specificity']:.4f}")

    return {
        "status": "ok",
        "n_samples": int(len(y_true)),
        "overall": overall,
        "ci": ci,
        "per_source": per_source,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    device = CFG.device()
    print(f"[ext_eval] device = {device}")

    # find all fold checkpoints
    ckpt_paths = sorted(CFG.CHECKPOINT_DIR.glob("fold_*_best.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No fold checkpoints found in {CFG.CHECKPOINT_DIR}. "
            "Run finetune (or run_parallel_folds) first.")
    print(f"[ext_eval] Found {len(ckpt_paths)} fold checkpoints: "
          f"{[p.stem for p in ckpt_paths]}")
    ckpt_paths = [str(p) for p in ckpt_paths]

    # load training subjects to exclude from external validation
    print("\n[ext_eval] Loading training subjects for exclusion ...")
    _, train_meta = load_cache("paired_binary")
    # Only exclude subjects from CinC2016 training-a (not EPHNOGRAM)
    cinc_subjects = set(
        train_meta[train_meta["source"] == "cinc2016"]["subject"].dropna().unique()
    )
    print(f"[ext_eval] CinC2016 training-a subjects to exclude: {len(cinc_subjects)}")

    results = {}

    # ------------------------------------------------------------------ PCG external
    print("\n" + "="*60)
    print("  PCG EXTERNAL VALIDATION")
    print("="*60)

    # CirCor DigiScope
    results["circor"] = ensemble_evaluate_with_filter(
        ckpt_paths, "pcg_binary", device,
        include_sources=["circor"],
        exclude_subjects=cinc_subjects,
        batch_size=args.batch,
        desc="CirCor DigiScope (PCG-only, paediatric)",
    )

    # CinC2016 training-b/c/d/e/f (exclude training-a subjects)
    results["cinc2016_external"] = ensemble_evaluate_with_filter(
        ckpt_paths, "pcg_binary", device,
        include_sources=["cinc2016"],
        exclude_subjects=cinc_subjects,
        batch_size=args.batch,
        desc="CinC2016 training-b/c/d/e/f (PCG-only)",
    )

    # ------------------------------------------------------------------ ECG external
    print("\n" + "="*60)
    print("  ECG EXTERNAL VALIDATION")
    print("="*60)

    results["chapman"] = ensemble_evaluate_with_filter(
        ckpt_paths, "ecg_binary", device,
        include_sources=["chapman"],
        batch_size=args.batch,
        desc="Chapman-Shaoxing 12-lead ECG",
    )

    results["ptbxl"] = ensemble_evaluate_with_filter(
        ckpt_paths, "ecg_binary", device,
        include_sources=["ptbxl"],
        batch_size=args.batch,
        desc="PTB-XL 12-lead ECG",
    )

    results["mitbih"] = ensemble_evaluate_with_filter(
        ckpt_paths, "ecg_binary", device,
        include_sources=["mitbih"],
        batch_size=args.batch,
        desc="MIT-BIH Arrhythmia Database",
    )

    results["cpsc2018"] = ensemble_evaluate_with_filter(
        ckpt_paths, "ecg_binary", device,
        include_sources=["c2020_cpsc_2018", "c2020_cpsc_2018_extra"],
        batch_size=args.batch,
        desc="CPSC-2018 12-lead ECG Challenge",
    )

    # ------------------------------------------------------------------ save
    out_path = CFG.RESULTS_DIR / "external_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ext_eval] Saved -> {out_path}")

    # Summary table
    print("\n" + "="*60)
    print("  SUMMARY — External Validation")
    print("="*60)
    print(f"{'Dataset':<35}  {'N':>8}  {'AUROC':>7}  {'F1':>7}  {'Sens':>7}  {'Spec':>7}")
    print("-" * 80)
    for name, res in results.items():
        if res.get("status") == "ok":
            m = res["overall"]
            print(f"{name:<35}  {res['n_samples']:>8,}  "
                  f"{m['auroc']:>7.4f}  {m['f1']:>7.4f}  "
                  f"{m['sensitivity']:>7.4f}  {m['specificity']:>7.4f}")
        else:
            print(f"{name:<35}  SKIPPED ({res.get('status')})")

    print("\n[ext_eval] DONE.")


if __name__ == "__main__":
    main()
