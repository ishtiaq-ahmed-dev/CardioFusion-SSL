"""CardioFusion-SSL fine-tuning v2 (Tier 2 + Tier 3 improvements).

Adds on top of scripts/finetune.py:
  - SpecAugment on PCG mel                (utils/augment.py)
  - MixUp on paired ECG+PCG               (utils/augment.py)
  - Modality dropout at 15%               (utils/augment.py)
  - ECG shift + amplitude jitter          (utils/augment.py)
  - SWA (Stochastic Weight Averaging)     (torch.optim.swa_utils)
  - Multi-seed ensembling per fold        (--seeds 3 by default)
  - Youden-optimal threshold reporting    (already in finetune.py compute_metrics)
  - Longer training window                (default 100 epochs, patience 20)

Usage:
    python -m scripts.finetune_v2 --ssl-ckpt checkpoints/ssl_pretrain_v2.pt \
        --epochs 100 --seeds 3 --folds 10

Every seed produces a separate checkpoint:
    checkpoints_v2/fold_{fold}_seed_{seed}_best.pt

Ensemble at eval time is the soft-vote across all folds × seeds.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader

from configs import CFG
from models.full_model import CardioFusionSSL
from data.adapter import (
    load_cache, PairedCacheDataset, collate_with_strings,
    subject_disjoint_kfold, build_weighted_sampler,
)
from utils.augment import AugCfg, augment_batch, soft_focal_loss
from scripts.finetune import (
    FocalLoss, EMA, cosine_lr, seed_all, compute_metrics,
    bootstrap_ci, evaluate_loader,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Per-fold, per-seed training
# ═══════════════════════════════════════════════════════════════════════════
def train_fold_seed(
    fold_idx: int, seed: int, train_idx, val_idx, test_idx, meta, args, device
) -> dict:
    """Train one (fold, seed) model with all v2 augmentations + SWA."""

    seed_all(CFG.SEED + seed * 10007 + fold_idx)   # deterministic but seed-varying

    print(f"\n{'=' * 60}\n  FOLD {fold_idx+1}  SEED {seed}\n{'=' * 60}", flush=True)

    # ── data loaders ────────────────────────────────────────────────────
    train_ds = PairedCacheDataset(args.cache, indices=train_idx)
    val_ds   = PairedCacheDataset(args.cache, indices=val_idx)
    test_ds  = PairedCacheDataset(args.cache, indices=test_idx)

    sampler = build_weighted_sampler(meta, train_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True,
                              drop_last=True, collate_fn=collate_with_strings)
    val_loader   = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_with_strings)
    test_loader  = DataLoader(test_ds, batch_size=args.batch * 2, shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_with_strings)

    # ── model ───────────────────────────────────────────────────────────
    model = CardioFusionSSL(enable_ssl=False, enable_recon=False).to(device)
    if args.ssl_ckpt and os.path.exists(args.ssl_ckpt):
        sd = torch.load(args.ssl_ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  loaded SSL ckpt: {len(missing)} missing, {len(unexpected)} unexpected keys")

    # ── class weights ───────────────────────────────────────────────────
    train_labels = meta.loc[train_idx, "binary"].astype(int).values
    counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    cw = 1.0 / counts
    cw = cw / cw.sum() * len(cw)
    class_weight = torch.tensor(cw, device=device)

    # ── augmentation config ─────────────────────────────────────────────
    aug_cfg = AugCfg()

    # ── optimiser + schedulers ──────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=CFG.WEIGHT_DECAY)
    scaler = GradScaler("cuda", enabled=(device == "cuda" and CFG.AMP))
    ema = EMA(model)

    # SWA
    swa_start_epoch = int(args.epochs * CFG.SWA_START_FRAC)
    swa_model = AveragedModel(model)
    swa_sched = SWALR(opt, swa_lr=CFG.SWA_LR)

    total_steps = args.epochs * max(1, len(train_loader))
    warmup = int(CFG.WARMUP_FRAC * total_steps)
    step = 0

    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None
    swa_started = False

    # ── training loop ───────────────────────────────────────────────────
    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss = []
        for batch in train_loader:
            for k in ("ecg", "pcg_mel", "has_ecg", "has_pcg", "label"):
                batch[k] = batch[k].to(device, non_blocking=True)

            # ── APPLY AUGMENTATIONS ──
            batch, soft_labels = augment_batch(batch, aug_cfg, n_classes=CFG.N_BINARY)

            # LR schedule (only during warmup / cosine phase, before SWA takes over)
            if ep <= swa_start_epoch:
                for g in opt.param_groups:
                    g["lr"] = cosine_lr(step, total_steps, args.lr, warmup)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device == "cuda" and CFG.AMP)):
                out = model(batch, mode="supervised")
                loss = soft_focal_loss(
                    out["logits"], soft_labels,
                    weight=class_weight, gamma=CFG.FOCAL_GAMMA,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            ema.update()
            ep_loss.append(loss.item())
            step += 1

        # SWA weight update
        if ep >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_sched.step()
            if not swa_started:
                print(f"  SWA started at epoch {ep}", flush=True)
                swa_started = True

        # validation with EMA weights (until SWA kicks in)
        if ep < swa_start_epoch:
            ema.apply()
            val_m = evaluate_loader(model, val_loader, device)
            ema.restore()
        else:
            # SWA phase — validate the SWA-averaged model
            update_bn(train_loader, swa_model, device=device)
            val_m = evaluate_loader(swa_model, val_loader, device)

        print(f"  f{fold_idx+1}s{seed} ep{ep:03d}  "
              f"loss={np.mean(ep_loss):.4f}  "
              f"val_f1={val_m['f1']:.4f}  val_auroc={val_m['auroc']:.4f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}",
              flush=True)

        # early stop based on val F1
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            patience_counter = 0
            if ep < swa_start_epoch:
                ema.apply()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                ema.restore()
            else:
                best_state = {k: v.cpu().clone() for k, v in swa_model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  early stop at epoch {ep}", flush=True)
                break

    # ── save checkpoint ─────────────────────────────────────────────────
    ckpt_dir = PROJECT_ROOT / "checkpoints_v2"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"fold_{fold_idx+1}_seed_{seed}_best.pt"
    torch.save(best_state, ckpt_path)
    print(f"  checkpoint -> {ckpt_path}", flush=True)

    # ── final test ──────────────────────────────────────────────────────
    if best_state is not None and swa_started and "n_averaged" in best_state:
        # SWA state — load into swa_model wrapper
        swa_model.load_state_dict(best_state)
        eval_model = swa_model
    else:
        model.load_state_dict(best_state)
        eval_model = model
    eval_model.eval()
    test_m, y_true, y_pred, y_prob = evaluate_loader(
        eval_model, test_loader, device, return_arrays=True)
    ci = bootstrap_ci(y_true, y_pred, y_prob)

    print(f"\n  FOLD {fold_idx+1} SEED {seed} TEST:  "
          f"acc={test_m['accuracy']:.4f}  f1={test_m['f1']:.4f}  "
          f"sens={test_m['sensitivity']:.4f}  spec={test_m['specificity']:.4f}  "
          f"auroc={test_m['auroc']:.4f}",
          flush=True)

    # cleanup
    del model, swa_model, opt, scaler, ema, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "fold": fold_idx + 1, "seed": seed,
        "metrics": test_m, "ci": ci,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="paired_binary")
    ap.add_argument("--ssl-ckpt", default=str(CFG.CHECKPOINT_DIR / "ssl_pretrain_v2.pt"))
    ap.add_argument("--folds", type=int, default=CFG.N_FOLDS)
    ap.add_argument("--seeds", type=int, default=3, help="Random seeds per fold")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=CFG.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=CFG.LR)
    ap.add_argument("--fold-only", type=int, default=None,
                    help="Run only this fold (0-indexed), for parallel launches")
    args = ap.parse_args()

    seed_all(CFG.SEED)
    device = CFG.device()
    print(f"[finetune_v2] device = {device}")
    print(f"[finetune_v2] {args.folds} folds × {args.seeds} seeds = "
          f"{args.folds * args.seeds} models will be trained")

    # ── load cache ──────────────────────────────────────────────────────
    _, meta = load_cache(args.cache)
    if "source" in meta.columns:
        sup_sources = [s for s in meta["source"].unique() if "ephnogram" not in s.lower()]
        meta = meta[meta["source"].isin(sup_sources)].copy().reset_index(drop=True)
        print(f"[finetune_v2] Supervised-only filter: N = {len(meta)}, "
              f"{meta['subject'].nunique()} subjects")

    # ── splits ──────────────────────────────────────────────────────────
    folds = subject_disjoint_kfold(meta, n_folds=args.folds)

    # ── train all (fold, seed) pairs ────────────────────────────────────
    all_results = []
    fold_range = [args.fold_only] if args.fold_only is not None else range(args.folds)
    for fold_idx in fold_range:
        train_idx, test_idx = folds[fold_idx]
        # within-fold val split (15% of training windows)
        n_val = int(0.15 * len(train_idx))
        rng = np.random.RandomState(CFG.SEED + fold_idx)
        val_pool = rng.permutation(train_idx)
        val_idx = val_pool[:n_val]
        train_idx = val_pool[n_val:]

        for seed in range(args.seeds):
            res = train_fold_seed(fold_idx, seed, train_idx, val_idx, test_idx,
                                   meta, args, device)
            all_results.append(res)

    # ── per-fold save (single-fold parallel-launcher mode) ──────────────
    if args.fold_only is not None:
        fold_dir = CFG.RESULTS_DIR / "fold_results_v2"
        fold_dir.mkdir(parents=True, exist_ok=True)
        out_path = fold_dir / f"fold_{args.fold_only + 1}.json"
        with open(out_path, "w") as f:
            json.dump({"fold": args.fold_only + 1, "results": all_results}, f, indent=2)
        print(f"\n[finetune_v2] fold {args.fold_only + 1} results -> {out_path}")

    # ── aggregate (full run only) ───────────────────────────────────────
    if args.fold_only is None:
        agg = {}
        for m in all_results[0]["metrics"]:
            vals = [r["metrics"][m] for r in all_results]
            agg[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        summary = {
            "n_folds": args.folds,
            "n_seeds": args.seeds,
            "aggregated": agg,
            "per_fold_seed": all_results,
        }
        out_path = CFG.RESULTS_DIR / "finetune_v2_results.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[finetune_v2] Saved -> {out_path}")

        # print summary
        print(f"\n{'=' * 60}\n  SUMMARY across {len(all_results)} models "
              f"({args.folds} folds × {args.seeds} seeds)\n{'=' * 60}")
        for m in ("auroc", "f1", "sensitivity", "specificity", "accuracy", "mcc"):
            if m in agg:
                print(f"  {m:<15s}: {agg[m]['mean']:.4f} ± {agg[m]['std']:.4f}")


if __name__ == "__main__":
    main()
