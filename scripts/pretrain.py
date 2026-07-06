"""SSL pretraining entry point for CardioFusion-SSL.

Loads the `paired_binary` cache (PhysioNet 2016 training-a + EPHNOGRAM clips),
runs cross-modal contrastive learning on the paired ECG+PCG, optionally with
within-modality augmentation contrast and a light masked-reconstruction term.

Usage (from project root):

    python -m scripts.pretrain --epochs 100 --batch 128 --resume <ckpt>

The trained encoder weights are saved under checkpoints/ssl_pretrain_*.pt and
are loaded by scripts/finetune.py for supervised fine-tuning.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from configs import CFG
from models import CardioFusionSSL
from pretraining.contrastive import CrossModalContrastiveLoss
from data.adapter import build_dataloader, collate_with_strings, load_cache


# --------------------------------------------------------------------- utility
def seed_all(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, (total - warmup))
    return 0.5 * base * (1.0 + np.cos(np.pi * p))


# --------------------------------------------------------------------- train loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="paired_binary",
                    help="cache name to read (default: paired_binary)")
    ap.add_argument("--epochs", type=int, default=CFG.EPOCHS_SSL)
    ap.add_argument("--batch", type=int, default=CFG.BATCH_SIZE_LARGE)
    ap.add_argument("--lr", type=float, default=CFG.LR_SSL)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--save", type=str,
                    default=str(CFG.CHECKPOINT_DIR / "ssl_pretrain.pt"))
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    seed_all(CFG.SEED)
    device = CFG.device()
    print(f"[pretrain] device = {device}")

    # ------------- data
    print(f"[pretrain] loading cache '{args.cache}' ...")
    arrays, meta = load_cache(args.cache)
    n = len(meta)
    has_ecg_frac = float(meta.get("has_ecg", 1.0).mean()) if "has_ecg" in meta else 1.0
    has_pcg_frac = float(meta.get("has_pcg", 1.0).mean()) if "has_pcg" in meta else 1.0
    print(f"[pretrain] N = {n} segments; "
          f"has_ecg fraction = {has_ecg_frac:.2f}; has_pcg fraction = {has_pcg_frac:.2f}")

    # For SSL we want only segments where BOTH modalities exist
    if "has_ecg" in arrays and "has_pcg" in arrays:
        paired_mask = (arrays["has_ecg"] > 0.5) & (arrays["has_pcg"] > 0.5)
    else:
        paired_mask = np.ones(n, dtype=bool)
    paired_idx = np.where(paired_mask)[0]
    print(f"[pretrain] {paired_idx.size} paired segments will be used for cross-modal SSL")

    if paired_idx.size < 16:
        raise SystemExit("Too few paired segments for SSL. Build paired cache first.")

    loader = build_dataloader(
        args.cache, indices=paired_idx,
        batch_size=args.batch, shuffle=True,
        weighted=False, num_workers=CFG.NUM_WORKERS,
    )
    # patch collate to keep strings list-typed
    loader.collate_fn = collate_with_strings

    # ------------- model
    model = CardioFusionSSL(enable_ssl=True, enable_recon=False).to(device)
    counts = model.count_parameters()
    print(f"[pretrain] model params: total = {counts['total']:,}")

    if args.resume:
        sd = torch.load(args.resume, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(f"[pretrain] resumed from {args.resume}")

    crit = CrossModalContrastiveLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=CFG.WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = GradScaler("cuda", enabled=(device == "cuda" and CFG.AMP))

    total_steps = args.epochs * max(1, len(loader))
    warmup_steps = int(CFG.WARMUP_FRAC * total_steps)
    step = 0
    best_loss = float("inf")
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_losses = []
        ep_accs = []
        pbar = tqdm(loader, desc=f"epoch {ep:03d}/{args.epochs}", leave=False)
        for batch in pbar:
            for k in ("ecg", "pcg_mel", "has_ecg", "has_pcg"):
                batch[k] = batch[k].to(device, non_blocking=True)

            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total_steps, args.lr, warmup_steps)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device == "cuda" and CFG.AMP)):
                out = model(batch, mode="ssl")
                losses = crit(out["z_ecg"], out["z_pcg"])
                loss = losses["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            ep_losses.append(loss.item())
            ep_accs.append(losses["acc"]["top1_ab"])
            step += 1
            pbar.set_postfix(loss=f"{np.mean(ep_losses[-50:]):.3f}",
                             top1=f"{np.mean(ep_accs[-50:]):.2f}")

        ep_loss = float(np.mean(ep_losses))
        ep_top1 = float(np.mean(ep_accs))
        history.append({"epoch": ep, "loss": ep_loss, "top1": ep_top1,
                        "lr": opt.param_groups[0]["lr"]})
        print(f"[pretrain] epoch {ep:03d}  loss={ep_loss:.4f}  top1={ep_top1:.3f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}")

        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save(model.state_dict(), args.save)
            with open(args.save + ".meta.json", "w") as f:
                json.dump({"epoch": ep, "loss": ep_loss, "top1": ep_top1,
                           "config": {k: str(v) for k, v in vars(CFG).items()
                                      if not k.startswith("_")}}, f, indent=2)
            print(f"[pretrain] saved best -> {args.save}")

    # final history dump
    hist_path = CFG.RESULTS_DIR / "ssl_pretrain_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[pretrain] history -> {hist_path}")
    print(f"[pretrain] DONE.  best_loss = {best_loss:.4f}")


if __name__ == "__main__":
    main()
