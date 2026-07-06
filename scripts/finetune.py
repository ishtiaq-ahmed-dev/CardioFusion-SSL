"""Supervised fine-tuning for CardioFusion-SSL.

Loads a SSL-pretrained checkpoint (or trains from scratch if none provided),
runs subject-disjoint stratified 10-fold CV on the paired_binary cache, and
reports per-fold + aggregated metrics with bootstrap 95% CIs.

Usage:
    python -m scripts.finetune --ssl-ckpt checkpoints/ssl_pretrain.pt --epochs 80
    python -m scripts.finetune --from-scratch --epochs 100
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
from tqdm.auto import tqdm

from configs import CFG
from models.full_model import CardioFusionSSL
from data.adapter import (
    load_cache, PairedCacheDataset, build_dataloader, collate_with_strings,
    subject_disjoint_kfold, build_weighted_sampler,
)


# --------------------------------------------------------------------- reproducibility
def seed_all(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------- focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = CFG.FOCAL_GAMMA,
                 label_smoothing: float = CFG.LABEL_SMOOTHING,
                 weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# --------------------------------------------------------------------- EMA
class EMA:
    def __init__(self, model: nn.Module, decay: float = CFG.EMA_DECAY):
        self.model = model
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters()}

    def update(self):
        for n, p in self.model.named_parameters():
            self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self):
        self.backup = {n: p.data.clone() for n, p in self.model.named_parameters()}
        for n, p in self.model.named_parameters():
            p.data.copy_(self.shadow[n])

    def restore(self):
        for n, p in self.model.named_parameters():
            p.data.copy_(self.backup[n])


# --------------------------------------------------------------------- LR schedule
def cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, (total - warmup))
    return 0.5 * base * (1.0 + np.cos(np.pi * p))


# --------------------------------------------------------------------- metrics
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score, average_precision_score,
                                 matthews_corrcoef, confusion_matrix, roc_curve,
                                 balanced_accuracy_score)
    m = {}
    # ── default threshold (0.5 = argmax) ────────────────────────────────────
    m["accuracy"]    = accuracy_score(y_true, y_pred)
    m["bal_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    m["f1"]          = f1_score(y_true, y_pred, average="macro")
    m["precision"]   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    m["recall"]      = recall_score(y_true, y_pred, average="macro", zero_division=0)
    m["sensitivity"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    m["mcc"]         = matthews_corrcoef(y_true, y_pred)
    try:
        m["auroc"]   = roc_auc_score(y_true, y_prob)
    except ValueError:
        m["auroc"]   = 0.0
    try:
        m["auprc"]   = average_precision_score(y_true, y_prob)
    except ValueError:
        m["auprc"]   = 0.0
    # ── Youden-index optimal threshold ──────────────────────────────────────
    # Maximises sensitivity + specificity - 1 (threshold-independent operating point)
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden_idx = np.argmax(tpr - fpr)
        opt_thresh = float(thresholds[youden_idx])
        y_pred_opt = (y_prob >= opt_thresh).astype(int)
        tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_true, y_pred_opt, labels=[0,1]).ravel()
        m["youden_threshold"]    = opt_thresh
        m["youden_sensitivity"]  = tp_o / (tp_o + fn_o) if (tp_o + fn_o) > 0 else 0.0
        m["youden_specificity"]  = tn_o / (tn_o + fp_o) if (tn_o + fp_o) > 0 else 0.0
        m["youden_f1"]           = f1_score(y_true, y_pred_opt, average="macro")
        m["youden_accuracy"]     = accuracy_score(y_true, y_pred_opt)
        m["youden_mcc"]          = matthews_corrcoef(y_true, y_pred_opt)
    except Exception:
        pass
    return m


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                 n_boot: int = CFG.BOOTSTRAP_ITERS,
                 ci: float = CFG.BOOTSTRAP_CI) -> dict:
    rng = np.random.RandomState(CFG.SEED)
    n = len(y_true)
    records = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        records.append(compute_metrics(y_true[idx], y_pred[idx], y_prob[idx]))
    df = pd.DataFrame(records)
    lo = (1 - ci) / 2
    hi = 1 - lo
    out = {}
    for col in df.columns:
        out[col] = {
            "mean": float(df[col].mean()),
            "lo": float(df[col].quantile(lo)),
            "hi": float(df[col].quantile(hi)),
        }
    return out


# --------------------------------------------------------------------- single fold
def train_one_fold(fold_idx: int, train_idx: np.ndarray, test_idx: np.ndarray,
                   arrays: dict, meta: pd.DataFrame, args, device: str) -> dict:
    """Train + evaluate one fold. Returns metrics dict."""
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx + 1}/{args.n_folds}")
    print(f"  train={len(train_idx)} test={len(test_idx)} segments")
    print(f"{'='*60}")

    # split train into train/val (85/15)
    rng = np.random.RandomState(CFG.SEED + fold_idx)
    train_idx = train_idx.copy()
    rng.shuffle(train_idx)
    n_val = int(len(train_idx) * CFG.VAL_FRAC)
    val_idx = train_idx[:n_val]
    train_idx = train_idx[n_val:]

    # data loaders
    from torch.utils.data import DataLoader
    train_ds = PairedCacheDataset(args.cache, indices=train_idx)
    val_ds   = PairedCacheDataset(args.cache, indices=val_idx)
    test_ds  = PairedCacheDataset(args.cache, indices=test_idx)

    sampler = build_weighted_sampler(meta, train_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True,
                              drop_last=True, collate_fn=collate_with_strings)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch * 2, shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_with_strings)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch * 2, shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_with_strings)

    # model
    model = CardioFusionSSL(enable_ssl=False, enable_recon=False).to(device)
    if args.ssl_ckpt and os.path.exists(args.ssl_ckpt):
        sd = torch.load(args.ssl_ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  loaded SSL ckpt: {len(missing)} missing, {len(unexpected)} unexpected keys")

    if getattr(args, "compile", False):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  [compile] torch.compile() active")
        except Exception as e:
            print(f"  [compile] skipped: {e}")

    # class weights
    train_labels = meta.loc[train_idx, "binary"].astype(int).values
    counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    cw = 1.0 / counts
    cw = cw / cw.sum() * len(cw)
    criterion = FocalLoss(weight=torch.tensor(cw, device=device))

    # optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=CFG.WEIGHT_DECAY)
    scaler = GradScaler("cuda", enabled=(device == "cuda" and CFG.AMP))
    ema = EMA(model)

    total_steps = args.epochs * max(1, len(train_loader))
    warmup = int(CFG.WARMUP_FRAC * total_steps)
    step = 0
    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss = []
        for batch in train_loader:
            for k in ("ecg", "pcg_mel", "has_ecg", "has_pcg", "label"):
                batch[k] = batch[k].to(device, non_blocking=True)

            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total_steps, args.lr, warmup)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device == "cuda" and CFG.AMP)):
                out = model(batch, mode="supervised")
                loss = criterion(out["logits"], batch["label"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            ema.update()
            ep_loss.append(loss.item())
            step += 1

        # validation with EMA weights
        ema.apply()
        val_m = evaluate_loader(model, val_loader, device)
        ema.restore()

        print(f"  fold {fold_idx+1} ep {ep:03d}  "
              f"loss={np.mean(ep_loss):.4f}  "
              f"val_acc={val_m['accuracy']:.4f}  val_f1={val_m['f1']:.4f}  "
              f"val_auroc={val_m['auroc']:.4f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            patience_counter = 0
            ema.apply()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ema.restore()
        else:
            patience_counter += 1
            if patience_counter >= CFG.EARLY_STOP_PATIENCE:
                print(f"  early stop at epoch {ep}")
                break

    # save fold checkpoint before evaluation (safe against crash during metrics)
    ckpt_path = CFG.CHECKPOINT_DIR / f"fold_{fold_idx+1}_best.pt"
    torch.save(best_state, ckpt_path)
    print(f"  checkpoint -> {ckpt_path}", flush=True)

    # final test with best EMA weights
    model.load_state_dict(best_state)
    model.eval()
    test_m, y_true, y_pred, y_prob = evaluate_loader(model, test_loader, device,
                                                      return_arrays=True)
    ci = bootstrap_ci(y_true, y_pred, y_prob)

    print(f"\n  FOLD {fold_idx+1} TEST:  acc={test_m['accuracy']:.4f}  "
          f"f1={test_m['f1']:.4f}  sens={test_m['sensitivity']:.4f}  "
          f"spec={test_m['specificity']:.4f}  auroc={test_m['auroc']:.4f}  "
          f"mcc={test_m['mcc']:.4f}", flush=True)
    if "youden_sensitivity" in test_m:
        print(f"  FOLD {fold_idx+1} YOUDEN (thr={test_m['youden_threshold']:.3f}):  "
              f"sens={test_m['youden_sensitivity']:.4f}  "
              f"spec={test_m['youden_specificity']:.4f}  "
              f"f1={test_m['youden_f1']:.4f}  "
              f"acc={test_m['youden_accuracy']:.4f}", flush=True)

    # cleanup
    del model, opt, scaler, ema, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return {"fold": fold_idx + 1, "metrics": test_m, "ci": ci,
            "y_true": y_true.tolist(), "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist()}


# --------------------------------------------------------------------- eval helper
def evaluate_loader(model, loader, device, return_arrays=False):
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
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
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    m = compute_metrics(y_true, y_pred, y_prob)
    if return_arrays:
        return m, y_true, y_pred, y_prob
    return m


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="paired_binary")
    ap.add_argument("--ssl-ckpt", default=str(CFG.CHECKPOINT_DIR / "ssl_pretrain.pt"))
    ap.add_argument("--from-scratch", action="store_true")
    ap.add_argument("--epochs", type=int, default=CFG.EPOCHS_FINETUNE)
    ap.add_argument("--batch", type=int, default=CFG.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=CFG.LR)
    ap.add_argument("--n-folds", type=int, default=CFG.N_FOLDS)
    ap.add_argument("--seed", type=int, default=CFG.SEED)
    ap.add_argument("--fold-indices", default="all",
                    help="Comma-separated 1-indexed fold numbers (e.g. '3,4,5') or 'all'")
    ap.add_argument("--dl-workers", type=int, default=None,
                    help="DataLoader num_workers override (default: CFG.NUM_WORKERS)")
    ap.add_argument("--compile", action="store_true",
                    help="Apply torch.compile(mode='reduce-overhead') to the model")
    args = ap.parse_args()

    if args.from_scratch:
        args.ssl_ckpt = None

    # DataLoader worker override
    if args.dl_workers is not None:
        CFG.NUM_WORKERS = args.dl_workers

    # Parse fold indices (1-indexed → 0-indexed internally)
    if args.fold_indices == "all":
        fold_indices_to_run = list(range(args.n_folds))
    else:
        fold_indices_to_run = [int(x.strip()) - 1 for x in args.fold_indices.split(",")]

    seed_all(args.seed)
    device = CFG.device()
    print(f"[finetune] device = {device}")
    print(f"[finetune] ssl_ckpt = {args.ssl_ckpt}")
    print(f"[finetune] epochs = {args.epochs}, batch = {args.batch}, lr = {args.lr}")
    print(f"[finetune] n_folds = {args.n_folds}, running folds: {[i+1 for i in fold_indices_to_run]}")
    print(f"[finetune] dl_workers = {CFG.NUM_WORKERS}, compile = {args.compile}")

    # load cache
    arrays, meta = load_cache(args.cache)
    print(f"[finetune] N total = {len(meta)} segments, "
          f"{meta['subject'].nunique()} subjects")

    # Filter to supervised-only sources (exclude healthy-only SSL corpora like EPHNOGRAM)
    if "source" in meta.columns:
        sup_sources = [s for s in meta["source"].unique() if "ephnogram" not in s.lower()]
        meta = meta[meta["source"].isin(sup_sources)].copy()
        print(f"[finetune] After source filter ({sup_sources}): N = {len(meta)}, "
              f"{meta['subject'].nunique()} subjects")

    print(f"[finetune] label distribution: {meta['binary'].value_counts().to_dict()}")

    # k-fold — must build ALL folds so indices are stable across parallel runs
    folds = subject_disjoint_kfold(meta, n_folds=args.n_folds, seed=args.seed)
    print(f"[finetune] {len(folds)} folds generated")

    # Per-fold result directory for parallel aggregation
    fold_result_dir = CFG.RESULTS_DIR / "fold_results"
    fold_result_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for i in fold_indices_to_run:
        train_idx, test_idx = folds[i]
        result = train_one_fold(i, train_idx, test_idx, arrays, meta, args, device)
        all_results.append(result)

        # Save per-fold JSON immediately for parallel collection
        fold_json = fold_result_dir / f"fold_{i+1}.json"
        with open(fold_json, "w") as f_out:
            json.dump(result, f_out, indent=2)
        print(f"[finetune] fold {i+1} result -> {fold_json}")

    # aggregate
    metric_names = list(all_results[0]["metrics"].keys())
    agg = {}
    for m in metric_names:
        vals = [r["metrics"][m] for r in all_results]
        agg[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                  "per_fold": vals}

    print(f"\n{'='*60}")
    print(f"  AGGREGATED {args.n_folds}-FOLD RESULTS")
    print(f"{'='*60}")
    for m in metric_names:
        print(f"  {m:<15s} = {agg[m]['mean']:.4f} ± {agg[m]['std']:.4f}")

    # save
    results_path = CFG.RESULTS_DIR / "finetune_results.json"
    with open(results_path, "w") as f:
        json.dump({"aggregated": agg, "per_fold": all_results,
                   "config": {"epochs": args.epochs, "batch": args.batch,
                              "lr": args.lr, "n_folds": args.n_folds,
                              "ssl_ckpt": args.ssl_ckpt, "seed": args.seed}}, f, indent=2)
    print(f"[finetune] results -> {results_path}")

    # CSV summary table
    rows = []
    for r in all_results:
        row = {"fold": r["fold"]}
        row.update(r["metrics"])
        rows.append(row)
    row_agg = {"fold": "mean±std"}
    for m in metric_names:
        row_agg[m] = f"{agg[m]['mean']:.4f}±{agg[m]['std']:.4f}"
    rows.append(row_agg)
    df = pd.DataFrame(rows)
    csv_path = CFG.RESULTS_DIR / "tables" / "kfold_results.csv"
    os.makedirs(csv_path.parent, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[finetune] CSV table -> {csv_path}")
    print(df.to_string(index=False))

    print(f"\n[finetune] DONE.")


if __name__ == "__main__":
    main()
