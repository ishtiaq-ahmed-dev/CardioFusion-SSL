"""Test-time augmentation (TTA) on the v2 internal 10-fold CV.

Reloads each of the 10 fold-seed checkpoints and reruns inference on its
subject-disjoint test set with N augmented views per window. Averages the
predictions in logit space before applying the sigmoid.

TTA views: original + 2 shifted/amplitude-jittered ECG variants.
No modality dropout at TTA (all inputs stay paired).
No SpecAugment at TTA (paper-style TTA uses mild variations only).

Wall clock: ~5 min per fold × 10 folds = ~50 min.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from configs import CFG
from models.full_model import CardioFusionSSL
from data.adapter import (
    load_cache, PairedCacheDataset, collate_with_strings,
    subject_disjoint_kfold,
)
from scripts.finetune import compute_metrics, seed_all

CKPT_DIR = PROJECT_ROOT / "checkpoints_v2"


# ═══════════════════════════════════════════════════════════════════════════
#  TTA views
# ═══════════════════════════════════════════════════════════════════════════
def _tta_view(batch: dict, view_idx: int, device: str) -> dict:
    """Return an augmented view of the batch. view_idx=0 => original."""
    if view_idx == 0:
        return batch

    B = batch["ecg"].size(0)
    new_batch = {**batch}

    if view_idx == 1:
        # Small forward time shift + mild amplitude gain
        shifts = torch.full((B,), 30, device=device)                 # +30 samples = +60 ms
        gain = 1.05
    elif view_idx == 2:
        # Backward shift + slightly lower amplitude
        shifts = torch.full((B,), -30, device=device)
        gain = 0.95
    else:
        return batch

    ecg = batch["ecg"].clone()
    for b in range(B):
        s = int(shifts[b].item())
        if s != 0:
            ecg[b] = torch.roll(batch["ecg"][b], shifts=s, dims=-1)
    ecg = ecg * gain
    new_batch["ecg"] = ecg
    return new_batch


def _infer_with_tta(model, loader, device, n_views: int = 3):
    model.eval()
    all_true = []
    logits_sum = []
    with torch.no_grad():
        for batch in loader:
            for k in ("ecg", "pcg_mel", "has_ecg", "has_pcg", "label"):
                batch[k] = batch[k].to(device, non_blocking=True)

            # Accumulate logits across TTA views
            view_logits = []
            for v in range(n_views):
                v_batch = _tta_view(batch, v, device)
                with autocast("cuda", enabled=(device == "cuda" and CFG.AMP)):
                    out = model(v_batch, mode="supervised")
                # Extract positive-class logit
                logits = out["logits"]
                pos_logit = logits[:, 1] - logits[:, 0]
                view_logits.append(pos_logit)
            avg_pos_logit = torch.stack(view_logits, dim=0).mean(dim=0)
            prob = torch.sigmoid(avg_pos_logit).cpu().numpy()

            logits_sum.append(prob)
            all_true.append(batch["label"].cpu().numpy())

    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(logits_sum)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    return metrics, y_true, y_pred, y_prob


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    seed_all(CFG.SEED)
    device = CFG.device()
    print(f"[tta] device = {device}")

    _, meta = load_cache("paired_binary")
    if "source" in meta.columns:
        sup = [s for s in meta["source"].unique() if "ephnogram" not in s.lower()]
        meta = meta[meta["source"].isin(sup)].copy().reset_index(drop=True)

    folds = subject_disjoint_kfold(meta, n_folds=10)
    print(f"[tta] {len(folds)} folds")

    all_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        ckpt_path = CKPT_DIR / f"fold_{fold_idx + 1}_seed_0_best.pt"
        if not ckpt_path.exists():
            print(f"[tta] fold {fold_idx + 1}: no checkpoint at {ckpt_path}; skipping")
            continue

        test_ds = PairedCacheDataset("paired_binary", indices=test_idx)
        loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                            num_workers=0, pin_memory=True,
                            collate_fn=collate_with_strings)

        model = CardioFusionSSL(enable_ssl=False, enable_recon=False).to(device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        # SWA checkpoints have an "n_averaged" entry — strip module. prefix if present
        clean_sd = {}
        for k, v in sd.items():
            if k == "n_averaged":
                continue
            if k.startswith("module."):
                clean_sd[k[len("module."):]] = v
            else:
                clean_sd[k] = v
        missing, unexpected = model.load_state_dict(clean_sd, strict=False)
        if missing:
            print(f"  fold {fold_idx + 1}: {len(missing)} missing keys ({missing[:3]}...)")

        metrics, y_true, y_pred, y_prob = _infer_with_tta(model, loader, device, n_views=3)

        print(f"  fold {fold_idx + 1:2d}: N={len(y_true)}  auroc={metrics['auroc']:.4f}  "
              f"f1={metrics['f1']:.4f}  sens={metrics['sensitivity']:.4f}  "
              f"spec={metrics['specificity']:.4f}", flush=True)

        all_results.append({
            "fold": fold_idx + 1,
            "metrics": metrics,
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist(),
        })

        del model
        torch.cuda.empty_cache()

    # Aggregate
    agg = {}
    for m in all_results[0]["metrics"]:
        vals = [r["metrics"][m] for r in all_results]
        agg[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    print(f"\n{'=' * 60}\n  SUMMARY v2 + TTA (3 views)\n{'=' * 60}")
    for m in ("auroc", "f1", "sensitivity", "specificity", "accuracy", "mcc"):
        if m in agg:
            print(f"  {m:<15s}: {agg[m]['mean']:.4f} ± {agg[m]['std']:.4f}")

    out_path = CFG.RESULTS_DIR / "finetune_v2_tta_results.json"
    with open(out_path, "w") as f:
        json.dump({"aggregated": agg, "per_fold": all_results}, f, indent=2)
    print(f"\n[tta] saved -> {out_path}")


if __name__ == "__main__":
    main()
