"""Cross-dataset external validation for CardioFusion-SSL.

Loads fold checkpoints from finetune, runs inference on external caches
(pcg_binary, ecg_binary, disease), and reports per-dataset metrics with
bootstrap CIs. This is the "generalization story" — the central novelty.

Usage:
    python -m scripts.evaluate
    python -m scripts.evaluate --ckpt checkpoints/fold_1_best.pt --datasets pcg_binary ecg_binary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs import CFG
from models.full_model import CardioFusionSSL
from data.adapter import (
    PairedCacheDataset, collate_with_strings, load_cache,
)
from scripts.finetune import compute_metrics, bootstrap_ci


# --------------------------------------------------------------------- eval
def evaluate_cache(model, cache_name: str, device: str, batch_size: int = 64,
                   max_samples: int = 50000, seed: int = 42,
                   include_sources: list[str] | None = None,
                   exclude_subjects: set | None = None) -> dict:
    """Evaluate a model on an entire cache. Returns metrics + CIs.

    max_samples: cap at this many total samples. 0 = no limit.
    include_sources: if set, only evaluate rows where meta["source"] is in this list.
    exclude_subjects: if set, exclude rows where meta["subject"] is in this set.
    """
    print(f"\n  evaluating on '{cache_name}' ...")
    try:
        arrays, meta = load_cache(cache_name)
    except FileNotFoundError:
        print(f"    SKIP: cache '{cache_name}' not found")
        return {"status": "not_found"}

    if "binary" not in meta.columns:
        print(f"    SKIP: no binary labels in {cache_name}")
        return {"status": "no_labels"}

    # Source filter (e.g., include only circor for PCG external validation)
    mask = np.ones(len(meta), dtype=bool)
    if include_sources and "source" in meta.columns:
        mask &= meta["source"].isin(include_sources).values
        print(f"    source filter {include_sources}: {mask.sum()} / {len(meta)} rows kept")

    # Subject exclusion (e.g., exclude training-a subjects from pcg_binary)
    if exclude_subjects and "subject" in meta.columns:
        excl_mask = meta["subject"].isin(exclude_subjects).values
        mask &= ~excl_mask
        print(f"    excluded {excl_mask.sum()} rows from {len(exclude_subjects)} training subjects")

    base_indices = np.where(mask)[0]
    if len(base_indices) == 0:
        print(f"    SKIP: no samples after filtering")
        return {"status": "empty_after_filter"}

    # Sample indices if cache is very large
    indices = base_indices
    if max_samples and len(base_indices) > max_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(base_indices, size=max_samples, replace=False)
        indices = np.sort(indices)
        print(f"    sampling {max_samples} of {len(base_indices)} filtered samples")

    ds = PairedCacheDataset(cache_name, indices=indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True,
                        collate_fn=collate_with_strings)

    model.eval()
    all_true, all_pred, all_prob = [], [], []
    all_sources = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"    {cache_name}", leave=False):
            for k in ("ecg", "pcg_mel", "has_ecg", "has_pcg", "label"):
                batch[k] = batch[k].to(device, non_blocking=True)
            with autocast("cuda", enabled=(device == "cuda" and CFG.AMP)):
                out = model(batch, mode="supervised")
            prob = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
            pred = out["logits"].argmax(dim=-1).cpu().numpy()
            true = batch["label"].cpu().numpy()
            all_true.append(true)
            all_pred.append(pred)
            all_prob.append(prob)
            all_sources.extend(batch.get("source", ["unknown"] * len(true)))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)

    # filter out samples with label=-1
    valid = y_true >= 0
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    y_prob = y_prob[valid]

    if len(y_true) == 0:
        return {"status": "no_valid_labels"}

    overall = compute_metrics(y_true, y_pred, y_prob)
    ci = bootstrap_ci(y_true, y_pred, y_prob)

    # per-source breakdown
    sources_arr = np.array(all_sources)[valid] if len(all_sources) > 0 else None
    per_source = {}
    if sources_arr is not None:
        for src in np.unique(sources_arr):
            mask = sources_arr == src
            if mask.sum() > 10:
                per_source[src] = compute_metrics(y_true[mask], y_pred[mask], y_prob[mask])

    print(f"    N={len(y_true)}  acc={overall['accuracy']:.4f}  "
          f"f1={overall['f1']:.4f}  auroc={overall['auroc']:.4f}  "
          f"sens={overall['sensitivity']:.4f}  spec={overall['specificity']:.4f}")

    return {
        "status": "ok",
        "n_samples": int(len(y_true)),
        "overall": overall,
        "ci": ci,
        "per_source": per_source,
    }


# --------------------------------------------------------------------- ensemble eval
def evaluate_ensemble(ckpt_paths: list[str], cache_name: str,
                      device: str, batch_size: int = 64) -> dict:
    """Soft-voting ensemble across fold checkpoints."""
    print(f"\n  ENSEMBLE evaluation on '{cache_name}' ({len(ckpt_paths)} models) ...")
    try:
        arrays, meta = load_cache(cache_name)
    except FileNotFoundError:
        return {"status": "not_found"}

    if "binary" not in meta.columns:
        return {"status": "no_labels"}

    ds = PairedCacheDataset(cache_name)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True,
                        collate_fn=collate_with_strings)

    # collect probabilities from each model
    all_probs = []
    for ckpt_path in ckpt_paths:
        model = CardioFusionSSL(enable_ssl=False, enable_recon=False).to(device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(sd, strict=False)
        model.eval()

        fold_prob = []
        with torch.no_grad():
            for batch in loader:
                for k in ("ecg", "pcg_mel", "has_ecg", "has_pcg", "label"):
                    batch[k] = batch[k].to(device, non_blocking=True)
                with autocast("cuda", enabled=(device == "cuda" and CFG.AMP)):
                    out = model(batch, mode="supervised")
                prob = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
                fold_prob.append(prob)
        all_probs.append(np.concatenate(fold_prob))
        del model
        torch.cuda.empty_cache()

    # soft vote
    avg_prob = np.mean(all_probs, axis=0)
    avg_pred = (avg_prob >= 0.5).astype(int)

    # get true labels
    all_true = []
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_with_strings):
        all_true.append(batch["label"].numpy())
    y_true = np.concatenate(all_true)

    valid = y_true >= 0
    y_true = y_true[valid]
    y_pred = avg_pred[valid]
    y_prob = avg_prob[valid]

    overall = compute_metrics(y_true, y_pred, y_prob)
    ci = bootstrap_ci(y_true, y_pred, y_prob)

    print(f"    ENSEMBLE N={len(y_true)}  acc={overall['accuracy']:.4f}  "
          f"f1={overall['f1']:.4f}  auroc={overall['auroc']:.4f}")

    return {"status": "ok", "n_samples": int(len(y_true)),
            "overall": overall, "ci": ci}


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+",
                    default=None,
                    help="Checkpoint path(s). If None, auto-discovers fold_*_best.pt")
    ap.add_argument("--datasets", nargs="+",
                    default=["paired_binary", "pcg_binary", "ecg_binary"],
                    help="Cache names to evaluate on")
    ap.add_argument("--include-sources", nargs="+", default=None,
                    help="Only evaluate rows from these sources (e.g. circor for PCG-only external)")
    ap.add_argument("--exclude-training-subjects", action="store_true",
                    help="Exclude subjects that appear in the paired_binary (supervised) training cache")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-samples", type=int, default=50000,
                    help="Max samples per dataset (samples randomly, default 50000). "
                         "Use 0 for no limit (very slow for ecg_binary).")
    ap.add_argument("--ensemble", action="store_true",
                    help="Also run soft-voting ensemble across all checkpoints")
    args = ap.parse_args()

    device = CFG.device()
    print(f"[evaluate] device = {device}")

    # Build training subject exclusion set if requested
    exclude_subjects = None
    if args.exclude_training_subjects:
        _, train_meta = load_cache("paired_binary")
        if "subject" in train_meta.columns:
            exclude_subjects = set(train_meta["subject"].dropna().unique())
            print(f"[evaluate] will exclude {len(exclude_subjects)} training subjects")
        else:
            print("[evaluate] WARNING: paired_binary has no 'subject' column — cannot exclude")

    # find checkpoints
    if args.ckpt:
        ckpt_paths = args.ckpt
    else:
        ckpt_dir = CFG.CHECKPOINT_DIR
        ckpt_paths = sorted(ckpt_dir.glob("fold_*_best.pt"))
        if not ckpt_paths:
            # try ssl pretrain as fallback
            ssl_path = ckpt_dir / "ssl_pretrain.pt"
            if ssl_path.exists():
                ckpt_paths = [ssl_path]
            else:
                raise FileNotFoundError(
                    f"No checkpoints found in {ckpt_dir}. Run finetune first.")
        ckpt_paths = [str(p) for p in ckpt_paths]
    print(f"[evaluate] found {len(ckpt_paths)} checkpoints")

    all_results = {}

    # per-checkpoint evaluation
    for ckpt_path in ckpt_paths:
        ckpt_name = Path(ckpt_path).stem
        print(f"\n{'='*60}")
        print(f"  Checkpoint: {ckpt_name}")
        print(f"{'='*60}")

        model = CardioFusionSSL(enable_ssl=False, enable_recon=False).to(device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(sd, strict=False)
        model.eval()

        ckpt_results = {}
        for ds_name in args.datasets:
            ckpt_results[ds_name] = evaluate_cache(
                model, ds_name, device, args.batch,
                max_samples=args.max_samples,
                include_sources=args.include_sources,
                exclude_subjects=exclude_subjects)
        all_results[ckpt_name] = ckpt_results

        del model
        torch.cuda.empty_cache()

    # ensemble evaluation
    if args.ensemble and len(ckpt_paths) > 1:
        print(f"\n{'='*60}")
        print(f"  ENSEMBLE ({len(ckpt_paths)} models)")
        print(f"{'='*60}")
        ens_results = {}
        for ds_name in args.datasets:
            ens_results[ds_name] = evaluate_ensemble(
                ckpt_paths, ds_name, device, args.batch)
        all_results["ensemble"] = ens_results

    # save
    results_path = CFG.RESULTS_DIR / "external_validation.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[evaluate] results -> {results_path}")

    # summary table
    rows = []
    for ckpt_name, ds_dict in all_results.items():
        for ds_name, res in ds_dict.items():
            if res.get("status") != "ok":
                continue
            row = {"checkpoint": ckpt_name, "dataset": ds_name,
                   "N": res["n_samples"]}
            row.update(res["overall"])
            rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        csv_path = CFG.RESULTS_DIR / "tables" / "external_validation.csv"
        os.makedirs(csv_path.parent, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"[evaluate] CSV -> {csv_path}")
        print(df.to_string(index=False))

    print("\n[evaluate] DONE.")


if __name__ == "__main__":
    main()
