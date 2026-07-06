"""CardioFusion-SSL SSL pretraining v2 (Tier 2 + Tier 3).

Extensions over scripts/pretrain.py:
  - Masked reconstruction loss enabled       (enable_recon=True in model)
  - Extended training (200 epochs default)   (was 100)
  - Auto-attaches mel_target = pcg_mel to each batch so the recon head has ground truth

The reconstruction heads and full-model plumbing already support this — we just
switch the flag and add mel/ECG targets to the batch dict.

Usage:
    python -m scripts.pretrain_v2 --epochs 200
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
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from configs import CFG
from models.full_model import CardioFusionSSL
from data.adapter import build_dataloader, collate_with_strings, load_cache
from pretraining.contrastive import CrossModalContrastiveLoss


# ═══════════════════════════════════════════════════════════════════════════
#  LR schedule
# ═══════════════════════════════════════════════════════════════════════════
def cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + np.cos(np.pi * p))


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="paired_binary")
    ap.add_argument("--epochs", type=int, default=200,
                    help="SSL epochs (v2 default = 200, was 100 in v1)")
    ap.add_argument("--batch", type=int, default=CFG.BATCH_SIZE_LARGE)
    ap.add_argument("--lr", type=float, default=CFG.LR_SSL)
    ap.add_argument("--w-recon", type=float, default=CFG.SSL_LOSS_W_RECON,
                    help="Weight on masked-reconstruction loss")
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    device = CFG.device()
    print(f"[pretrain_v2] device = {device}")
    print(f"[pretrain_v2] {args.epochs} epochs, batch = {args.batch}, "
          f"recon_weight = {args.w_recon}")

    # ── data: paired only (contrastive requires paired positives) ────────
    arrays, meta = load_cache(args.cache)
    if "has_ecg" in arrays and "has_pcg" in arrays:
        paired_mask = (arrays["has_ecg"] > 0.5) & (arrays["has_pcg"] > 0.5)
    else:
        paired_mask = np.ones(len(meta), dtype=bool)
    paired_idx = np.where(paired_mask)[0]
    print(f"[pretrain_v2] {paired_idx.size} paired segments will be used for SSL")

    if paired_idx.size < 16:
        raise SystemExit("Too few paired segments for SSL.")

    loader = build_dataloader(
        args.cache, indices=paired_idx,
        batch_size=args.batch, shuffle=True,
        weighted=False, num_workers=CFG.NUM_WORKERS,
    )
    loader.collate_fn = collate_with_strings

    # ── model with recon enabled ─────────────────────────────────────────
    model = CardioFusionSSL(enable_ssl=True, enable_recon=True).to(device)
    counts = model.count_parameters()
    print(f"[pretrain_v2] model params: total = {counts['total']:,}")

    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(f"[pretrain_v2] resumed from {args.resume}")

    contrastive = CrossModalContrastiveLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=CFG.WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = GradScaler("cuda", enabled=(device == "cuda" and CFG.AMP))

    total_steps = args.epochs * max(1, len(loader))
    warmup = int(CFG.WARMUP_FRAC * total_steps)
    step = 0
    best_loss = float("inf")
    history = []

    ckpt_path = CFG.CHECKPOINT_DIR / "ssl_pretrain_v2.pt"

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_stats = {"loss": [], "loss_cross": [], "loss_recon_ecg": [], "loss_recon_pcg": [],
                    "top1_ab": [], "top1_ba": []}
        pbar = tqdm(loader, desc=f"epoch {ep:03d}/{args.epochs}", leave=False)
        for batch in pbar:
            for k in ("ecg", "pcg_mel", "has_ecg", "has_pcg"):
                batch[k] = batch[k].to(device, non_blocking=True)
            # ── add recon targets (same as inputs — masked-recon happens inside model) ──
            batch["ecg_target"] = batch["ecg"].clone()
            batch["mel_target"] = batch["pcg_mel"].clone()

            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total_steps, args.lr, warmup)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device == "cuda" and CFG.AMP)):
                out = model(batch, mode="ssl")

                # cross-modal contrastive loss
                cl = contrastive(out["z_ecg"], out["z_pcg"])
                loss_total = cl["loss"]

                # masked reconstruction — MSE against target signal
                if "ecg_recon" in out:
                    tgt = batch["ecg_target"]
                    # ecg_target is (B, 1, T); ecg_recon may be (B, T) or (B, 1, T)
                    if out["ecg_recon"].dim() == 2:
                        pred = out["ecg_recon"].unsqueeze(1)
                    else:
                        pred = out["ecg_recon"]
                    # match length via crop
                    L = min(pred.size(-1), tgt.size(-1))
                    loss_recon_e = F.mse_loss(pred[..., :L], tgt[..., :L])
                    loss_total = loss_total + args.w_recon * loss_recon_e
                    ep_stats["loss_recon_ecg"].append(float(loss_recon_e.item()))

                if "mel_recon" in out:
                    tgt = batch["mel_target"]
                    pred = out["mel_recon"]
                    if pred.dim() == 3:  # (B, M, T)
                        pred = pred.unsqueeze(1)
                    M = min(pred.size(-2), tgt.size(-2))
                    T = min(pred.size(-1), tgt.size(-1))
                    loss_recon_p = F.mse_loss(pred[..., :M, :T], tgt[..., :M, :T])
                    loss_total = loss_total + args.w_recon * loss_recon_p
                    ep_stats["loss_recon_pcg"].append(float(loss_recon_p.item()))

            scaler.scale(loss_total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            ep_stats["loss"].append(float(loss_total.item()))
            ep_stats["loss_cross"].append(float(cl["loss_cross"].item()))
            ep_stats["top1_ab"].append(cl["acc"]["top1_ab"])
            ep_stats["top1_ba"].append(cl["acc"]["top1_ba"])
            step += 1

        mean_loss = float(np.mean(ep_stats["loss"]))
        mean_cross = float(np.mean(ep_stats["loss_cross"]))
        mean_top1 = float(np.mean(ep_stats["top1_ab"] + ep_stats["top1_ba"]) / 2)

        recon_e = float(np.mean(ep_stats["loss_recon_ecg"])) if ep_stats["loss_recon_ecg"] else None
        recon_p = float(np.mean(ep_stats["loss_recon_pcg"])) if ep_stats["loss_recon_pcg"] else None
        history_row = {
            "epoch": ep,
            "loss": mean_loss,
            "loss_cross": mean_cross,
            "loss_recon_ecg": recon_e,
            "loss_recon_pcg": recon_p,
            "top1": mean_top1,
        }
        history.append(history_row)
        recon_e_str = f"  recon_e={recon_e:.3f}" if recon_e is not None else ""
        recon_p_str = f"  recon_p={recon_p:.3f}" if recon_p is not None else ""
        print(f"  epoch {ep:03d}/{args.epochs}  "
              f"total={mean_loss:.4f}  cross={mean_cross:.4f}  top1={mean_top1:.3f}"
              f"{recon_e_str}{recon_p_str}",
              flush=True)

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  saved best checkpoint (loss = {best_loss:.4f}) -> {ckpt_path}", flush=True)

    # save training history
    history_path = CFG.RESULTS_DIR / "ssl_pretrain_v2_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[pretrain_v2] history -> {history_path}")
    print(f"[pretrain_v2] best checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
