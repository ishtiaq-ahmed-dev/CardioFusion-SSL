"""Ablation study for CardioFusion-SSL.

Tests 6 variants by modifying the model or training procedure and measuring
the drop in AUROC/F1 on the paired_binary test folds. Designed to run after
finetune.py has produced fold checkpoints and a finetune_results.json.

Ablation variants:
  1. full_model         — our full system (baseline reference, uses finetune results)
  2. no_ssl             — from-scratch fine-tuning (no SSL pretraining)
  3. single_scale       — fusion at one scale only (scale=16, mid-level), not hierarchical
  4. ecg_only           — ECG modality only (PCG replaced by missing token always)
  5. pcg_only           — PCG modality only (ECG replaced by missing token always)
  6. early_fusion       — simple concatenation of pooled ECG+PCG instead of cross-attention

Usage:
    python -m scripts.ablation
    python -m scripts.ablation --folds 3 --epochs 30   # faster run
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
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

from configs import CFG
from models.full_model import CardioFusionSSL
from models.mamba_ecg import ECGEncoder
from models.ast_pcg import PCGEncoder
from data.adapter import (
    load_cache, PairedCacheDataset, collate_with_strings,
    subject_disjoint_kfold, build_weighted_sampler,
)
from scripts.finetune import (
    seed_all, FocalLoss, EMA, cosine_lr,
    compute_metrics, bootstrap_ci, evaluate_loader,
)


# --------------------------------------------------------------------- variant models
class _EarlyFusionModel(nn.Module):
    """Simple concat of ECG + PCG pooled embeddings — no cross-attention."""

    def __init__(self):
        super().__init__()
        self.ecg_enc = ECGEncoder()
        self.pcg_enc = PCGEncoder()
        in_dim = CFG.D_MODEL * 2
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(in_dim // 2, CFG.N_BINARY),
        )

    def forward(self, batch, mode="supervised"):
        e = self.ecg_enc(batch["ecg"]).mean(dim=1)
        p = self.pcg_enc(batch["pcg_mel"]).mean(dim=1)
        feat = torch.cat([e, p], dim=-1)
        return {"logits": self.classifier(feat), "features": feat}


class _SingleScaleFusionModel(CardioFusionSSL):
    """Hierarchical fusion replaced with single mid-scale (16 tokens)."""

    def __init__(self):
        super().__init__(enable_ssl=False, enable_recon=False)
        # replace fusion with single-scale version
        from models.hier_fusion import HierFusion
        self.fusion = HierFusion(scales=(16,), depth=CFG.FUSION_DEPTH)
        from models.full_model import _ClassifierHead
        self.classifier = _ClassifierHead(in_dim=self.fusion.out_dim)


# --------------------------------------------------------------------- one ablation variant
def run_variant(name: str, model_factory, ssl_ckpt: str | None,
                arrays, meta, args, device) -> dict:
    """Run k-fold CV for one ablation variant. Returns aggregated metrics."""
    print(f"\n{'='*60}")
    print(f"  ABLATION: {name}")
    print(f"{'='*60}")

    folds = subject_disjoint_kfold(meta, n_folds=args.folds, seed=CFG.SEED)
    fold_metrics: list = []
    all_y_true: list = []
    all_y_pred: list = []
    all_y_prob: list = []

    for fi, (train_idx, test_idx) in enumerate(folds):
        train_idx = train_idx.copy()
        rng = np.random.RandomState(CFG.SEED + fi)
        rng.shuffle(train_idx)
        n_val = int(len(train_idx) * CFG.VAL_FRAC)
        val_idx, train_idx = train_idx[:n_val], train_idx[n_val:]

        train_ds = PairedCacheDataset(args.cache, indices=train_idx)
        val_ds   = PairedCacheDataset(args.cache, indices=val_idx)
        test_ds  = PairedCacheDataset(args.cache, indices=test_idx)

        sampler = build_weighted_sampler(meta, train_idx)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                  num_workers=CFG.NUM_WORKERS, pin_memory=True,
                                  drop_last=True, collate_fn=collate_with_strings)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch * 2, shuffle=False,
                                  num_workers=CFG.NUM_WORKERS, collate_fn=collate_with_strings)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch * 2, shuffle=False,
                                  num_workers=CFG.NUM_WORKERS, collate_fn=collate_with_strings)

        model = model_factory().to(device)
        if ssl_ckpt and os.path.exists(ssl_ckpt):
            sd = torch.load(ssl_ckpt, map_location=device, weights_only=False)
            model_sd = model.state_dict()
            # Only load keys that exist in this model AND have matching shapes.
            # Ablation variants (e.g. single_scale) have different classifier dims
            # so those keys must be skipped to avoid RuntimeError on shape mismatch.
            compat_sd = {k: v for k, v in sd.items()
                         if k in model_sd and v.shape == model_sd[k].shape}
            model.load_state_dict(compat_sd, strict=False)

        train_labels = meta.loc[train_idx, "binary"].astype(int).values
        counts = np.bincount(train_labels, minlength=2).astype(np.float32)
        cw = (1.0 / counts) / (1.0 / counts).sum() * 2
        criterion = FocalLoss(weight=torch.tensor(cw, device=device))

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=CFG.WEIGHT_DECAY)
        ema = EMA(model)

        best_val_f1 = 0.0
        patience = 0
        best_state = None
        total_steps = args.epochs * max(1, len(train_loader))
        warmup = int(CFG.WARMUP_FRAC * total_steps)
        step = 0

        for ep in range(1, args.epochs + 1):
            model.train()
            for batch in train_loader:
                for k in ("ecg", "pcg_mel", "has_ecg", "has_pcg", "label"):
                    batch[k] = batch[k].to(device, non_blocking=True)

                # for single-modality ablations, force missing flags
                if name == "ecg_only":
                    batch["has_pcg"] = torch.zeros_like(batch["has_pcg"])
                elif name == "pcg_only":
                    batch["has_ecg"] = torch.zeros_like(batch["has_ecg"])

                for g in opt.param_groups:
                    g["lr"] = cosine_lr(step, total_steps, args.lr, warmup)
                opt.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=(device == "cuda" and CFG.AMP)):
                    out = model(batch, mode="supervised")
                    loss = criterion(out["logits"], batch["label"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
                opt.step()
                ema.update()
                step += 1

            ema.apply()
            val_m = evaluate_loader(model, val_loader, device)
            ema.restore()
            if val_m["f1"] > best_val_f1:
                best_val_f1 = val_m["f1"]
                patience = 0
                ema.apply()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                ema.restore()
            else:
                patience += 1
                if patience >= CFG.EARLY_STOP_PATIENCE:
                    break

        model.load_state_dict(best_state)
        test_m, y_true, y_pred, y_prob = evaluate_loader(
            model, test_loader, device, return_arrays=True)
        fold_metrics.append(test_m)
        all_y_true.append(y_true.tolist())
        all_y_pred.append(y_pred.tolist())
        all_y_prob.append(y_prob.tolist())
        print(f"  {name} fold {fi+1}  auroc={test_m['auroc']:.4f}  f1={test_m['f1']:.4f}  "
              f"acc={test_m['accuracy']:.4f}")
        del model, opt, ema, train_loader, val_loader
        torch.cuda.empty_cache()

    agg = {}
    for m in fold_metrics[0]:
        vals = [fm[m] for fm in fold_metrics]
        agg[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    print(f"  {name} MEAN  auroc={agg['auroc']['mean']:.4f}±{agg['auroc']['std']:.4f}  "
          f"f1={agg['f1']['mean']:.4f}±{agg['f1']['std']:.4f}")

    # flatten predictions across all folds for statistical tests
    flat_y_true = [v for fold in all_y_true for v in fold]
    flat_y_pred = [v for fold in all_y_pred for v in fold]
    flat_y_prob = [v for fold in all_y_prob for v in fold]
    # Store per-fold AUROC for Wilcoxon signed-rank test against full model
    per_fold_aurocs = [float(fm["auroc"]) for fm in fold_metrics]
    preds = {
        "y_true": flat_y_true,
        "y_pred": flat_y_pred,
        "y_prob": flat_y_prob,
        "per_fold_aurocs": per_fold_aurocs,
    }
    return agg, preds


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="paired_binary")
    ap.add_argument("--ssl-ckpt", default=str(CFG.CHECKPOINT_DIR / "ssl_pretrain.pt"))
    ap.add_argument("--folds", type=int, default=CFG.N_FOLDS)
    ap.add_argument("--epochs", type=int, default=CFG.EPOCHS_FINETUNE)
    ap.add_argument("--batch", type=int, default=CFG.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=CFG.LR)
    ap.add_argument("--variants", nargs="+",
                    default=["no_ssl", "single_scale", "ecg_only", "pcg_only", "early_fusion"],
                    help="Which variants to test")
    args = ap.parse_args()

    seed_all(CFG.SEED)
    device = CFG.device()
    print(f"[ablation] device = {device}")
    arrays, meta = load_cache(args.cache)

    # Filter to supervised-only sources (same as finetune.py)
    if "source" in meta.columns:
        sup_sources = [s for s in meta["source"].unique() if "ephnogram" not in s.lower()]
        meta = meta[meta["source"].isin(sup_sources)].copy()
        print(f"[ablation] Supervised-only filter: N = {len(meta)}, "
              f"{meta['subject'].nunique()} subjects")

    # --- variant registry
    VARIANTS = {
        "no_ssl":        (lambda: CardioFusionSSL(enable_ssl=False, enable_recon=False), None),
        "single_scale":  (lambda: _SingleScaleFusionModel(), args.ssl_ckpt),
        "ecg_only":      (lambda: CardioFusionSSL(enable_ssl=False, enable_recon=False), args.ssl_ckpt),
        "pcg_only":      (lambda: CardioFusionSSL(enable_ssl=False, enable_recon=False), args.ssl_ckpt),
        "early_fusion":  (lambda: _EarlyFusionModel(), args.ssl_ckpt),
    }

    # load full model results from finetune
    ft_path = CFG.RESULTS_DIR / "finetune_results.json"
    ablation_results = {}
    if ft_path.exists():
        ft_data = json.load(open(ft_path))
        ablation_results["full_model"] = {
            m: ft_data["aggregated"][m]["mean"]
            for m in ft_data["aggregated"]
        }
        print(f"[ablation] full_model results loaded from {ft_path}")
    else:
        print(f"[ablation] WARNING: {ft_path} not found; run finetune first")
        ablation_results["full_model"] = {}

    variant_preds: dict = {}  # name -> {y_true, y_pred, y_prob} for statistical tests

    for v in args.variants:
        if v not in VARIANTS:
            print(f"[ablation] unknown variant {v!r} — skipping")
            continue
        factory, ckpt = VARIANTS[v]
        agg, preds = run_variant(v, factory, ckpt, arrays, meta, args, device)
        ablation_results[v] = {m: agg[m]["mean"] for m in agg}
        variant_preds[v] = preds

    # save metrics + per-variant predictions for statistical tests
    out_path = CFG.RESULTS_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump({**ablation_results,
                   "per_variant_preds": variant_preds}, f, indent=2)
    print(f"\n[ablation] saved -> {out_path}")

    # run statistical tests vs full model
    ft_path = CFG.RESULTS_DIR / "finetune_results.json"
    if ft_path.exists() and variant_preds:
        print("\n[ablation] Running statistical tests (McNemar + DeLong) ...")
        try:
            from utils.stats import run_all_tests_from_files
            stat_res = run_all_tests_from_files(
                str(ft_path), str(out_path),
                alpha=0.05, n_comparisons=len(variant_preds))
            stat_out = CFG.RESULTS_DIR / "statistical_tests.json"
            with open(stat_out, "w") as f:
                json.dump(stat_res, f, indent=2)
            print(f"[ablation] statistical tests saved -> {stat_out}")
        except Exception as exc:
            print(f"[ablation] statistical tests failed: {exc}")

    # summary table
    import pandas as pd
    metrics_show = ["auroc", "f1", "accuracy", "sensitivity", "specificity"]
    rows = []
    for vname, res in ablation_results.items():
        row = {"variant": vname}
        for m in metrics_show:
            row[m] = round(res.get(m, float("nan")), 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    csv_path = CFG.RESULTS_DIR / "tables" / "ablation_results.csv"
    os.makedirs(csv_path.parent, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[ablation] CSV -> {csv_path}")

    # try to generate visualisation
    try:
        from utils.visualise import plot_ablation_bars
        plot_ablation_bars(ablation_results)
    except Exception as e:
        print(f"[ablation] vis failed: {e}")

    print("\n[ablation] DONE.")


if __name__ == "__main__":
    main()
