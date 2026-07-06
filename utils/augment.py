"""Training-time augmentations for CardioFusion-SSL fine-tuning.

Adds four regularisers on top of the base training loop:
  - SpecAugment on PCG mel spectrograms (time + frequency masking)
  - MixUp on paired ECG+PCG inputs (label mixing, alpha=0.2)
  - Random modality dropout — force missing-modality tokens to train hard
  - Random time-shift and amplitude jitter on ECG

All augmentations are batch-level, run on GPU tensors, and are OFF by default.
Enable per-batch via ``augment_batch(batch, cfg=aug_cfg)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class AugCfg:
    """Turn-key augmentation config. Pass to augment_batch()."""

    # SpecAugment on PCG mel
    specaug_prob: float = 0.7           # probability of applying at all
    specaug_time_masks: int = 2         # number of time masks
    specaug_time_width: int = 12        # max width of each (in mel frames)
    specaug_freq_masks: int = 2         # number of frequency masks
    specaug_freq_width: int = 16        # max width of each (in mel bins)

    # MixUp on paired ECG+PCG
    mixup_prob: float = 0.5
    mixup_alpha: float = 0.2            # Beta distribution alpha; 0.2 is a common choice

    # Modality dropout — training-time forcing of missing-modality tokens
    mod_dropout_prob: float = 0.15      # fraction of batches where one modality is nuked
    mod_dropout_ecg_bias: float = 0.5   # 0.5 = symmetric; higher = more likely to nuke ECG

    # ECG time-shift + amplitude jitter
    ecg_shift_prob: float = 0.5
    ecg_shift_max: int = 100            # max shift in samples (100 samples @ 500 Hz = 200 ms)
    ecg_amp_jitter: float = 0.10        # relative amplitude jitter, U(1 - x, 1 + x)


DEFAULT_AUG = AugCfg()


# ═══════════════════════════════════════════════════════════════════════════
#  SpecAugment (PCG mel)
# ═══════════════════════════════════════════════════════════════════════════
def specaugment(mel: torch.Tensor, cfg: AugCfg) -> torch.Tensor:
    """SpecAugment (Park et al., 2019) applied to a batch of mel spectrograms.

    mel : (B, 1, M, T) float — will be modified in-place safely (via clone).
    Returns the augmented tensor.
    """
    if torch.rand(1).item() > cfg.specaug_prob:
        return mel

    B, C, M, T = mel.shape
    out = mel.clone()
    device = mel.device

    for b in range(B):
        for _ in range(cfg.specaug_time_masks):
            t = int(torch.randint(0, cfg.specaug_time_width + 1, (1,), device=device).item())
            if t > 0:
                t0 = int(torch.randint(0, max(1, T - t), (1,), device=device).item())
                out[b, :, :, t0:t0 + t] = 0.0
        for _ in range(cfg.specaug_freq_masks):
            f = int(torch.randint(0, cfg.specaug_freq_width + 1, (1,), device=device).item())
            if f > 0:
                f0 = int(torch.randint(0, max(1, M - f), (1,), device=device).item())
                out[b, :, f0:f0 + f, :] = 0.0
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  MixUp on paired inputs
# ═══════════════════════════════════════════════════════════════════════════
def mixup_paired(
    batch: Dict[str, torch.Tensor],
    cfg: AugCfg,
    n_classes: int = 2,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """MixUp (Zhang et al., 2018) on paired ECG+PCG inputs.

    Returns:
      mixed_batch : same keys as input, with ecg / pcg_mel linearly interpolated
      soft_labels : (B, n_classes) — soft target for label-smoothing-aware loss
    """
    B = batch["ecg"].size(0)
    device = batch["ecg"].device

    # No mixup case — return one-hot labels
    if torch.rand(1).item() > cfg.mixup_prob or cfg.mixup_alpha <= 0:
        one_hot = F.one_hot(batch["label"], num_classes=n_classes).float()
        return batch, one_hot

    # Draw lambda from Beta distribution
    lam = float(torch.distributions.Beta(cfg.mixup_alpha, cfg.mixup_alpha)
                .sample().item())
    lam = max(lam, 1.0 - lam)   # keep original label dominant

    # Shuffle indices for pairing
    idx = torch.randperm(B, device=device)

    mixed = {k: v.clone() if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    mixed["ecg"]     = lam * batch["ecg"]     + (1 - lam) * batch["ecg"][idx]
    mixed["pcg_mel"] = lam * batch["pcg_mel"] + (1 - lam) * batch["pcg_mel"][idx]
    # has_ecg / has_pcg — OR-combine (if either sample has the modality, keep it)
    mixed["has_ecg"] = torch.maximum(batch["has_ecg"], batch["has_ecg"][idx])
    mixed["has_pcg"] = torch.maximum(batch["has_pcg"], batch["has_pcg"][idx])

    # Soft labels
    y_a = F.one_hot(batch["label"], num_classes=n_classes).float()
    y_b = F.one_hot(batch["label"][idx], num_classes=n_classes).float()
    soft_labels = lam * y_a + (1 - lam) * y_b
    return mixed, soft_labels


# ═══════════════════════════════════════════════════════════════════════════
#  Modality dropout
# ═══════════════════════════════════════════════════════════════════════════
def modality_dropout(batch: Dict[str, torch.Tensor], cfg: AugCfg) -> Dict[str, torch.Tensor]:
    """Randomly drop a modality across the whole batch during training.

    Forces missing-modality tokens (m_ecg, m_pcg) to train hard so they are
    useful at inference in single-modality deployment scenarios.
    """
    if torch.rand(1).item() > cfg.mod_dropout_prob:
        return batch

    # Pick which modality to nuke — biased toward ECG if mod_dropout_ecg_bias > 0.5
    drop_ecg = torch.rand(1).item() < cfg.mod_dropout_ecg_bias

    B = batch["ecg"].size(0)
    device = batch["ecg"].device

    if drop_ecg:
        batch = {**batch}
        batch["ecg"] = torch.zeros_like(batch["ecg"])
        batch["has_ecg"] = torch.zeros(B, device=device)
    else:
        batch = {**batch}
        batch["pcg_mel"] = torch.zeros_like(batch["pcg_mel"])
        batch["has_pcg"] = torch.zeros(B, device=device)
    return batch


# ═══════════════════════════════════════════════════════════════════════════
#  ECG time-shift + amplitude jitter
# ═══════════════════════════════════════════════════════════════════════════
def ecg_shift_jitter(ecg: torch.Tensor, cfg: AugCfg) -> torch.Tensor:
    """Random circular time shift + per-sample amplitude jitter for ECG."""
    if torch.rand(1).item() > cfg.ecg_shift_prob:
        return ecg

    B, C, T = ecg.shape
    device = ecg.device
    # per-sample shift
    shifts = torch.randint(-cfg.ecg_shift_max, cfg.ecg_shift_max + 1, (B,), device=device)
    out = ecg.clone()
    for b in range(B):
        s = int(shifts[b].item())
        if s != 0:
            out[b] = torch.roll(ecg[b], shifts=s, dims=-1)

    # amplitude jitter — one scalar per sample
    if cfg.ecg_amp_jitter > 0:
        lo = 1.0 - cfg.ecg_amp_jitter
        hi = 1.0 + cfg.ecg_amp_jitter
        gains = lo + (hi - lo) * torch.rand(B, 1, 1, device=device)
        out = out * gains
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  One-shot pipeline
# ═══════════════════════════════════════════════════════════════════════════
def augment_batch(
    batch: Dict[str, torch.Tensor],
    cfg: AugCfg = DEFAULT_AUG,
    n_classes: int = 2,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Apply all training-time augmentations. Order matters.

    Sequence:
      1. ECG shift + amplitude jitter    (signal-space, per-sample)
      2. PCG SpecAugment                 (spectrogram-space, per-sample)
      3. Modality dropout                (batch-level, all-or-nothing)
      4. MixUp on paired inputs          (batch-level, produces soft labels)
    """
    batch = {**batch}
    batch["ecg"]     = ecg_shift_jitter(batch["ecg"], cfg)
    batch["pcg_mel"] = specaugment(batch["pcg_mel"], cfg)
    batch = modality_dropout(batch, cfg)
    batch, soft_labels = mixup_paired(batch, cfg, n_classes=n_classes)
    return batch, soft_labels


# ═══════════════════════════════════════════════════════════════════════════
#  Soft-target focal loss (for MixUp compatibility)
# ═══════════════════════════════════════════════════════════════════════════
def soft_focal_loss(
    logits: torch.Tensor,
    soft_labels: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss that accepts soft (MixUp) targets.

    logits      : (B, C)
    soft_labels : (B, C) — probability distribution (rows sum to 1)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    # Focal weighting: (1 - p_t)^gamma with p_t = sum(soft_labels * probs)
    p_t = (soft_labels * probs).sum(dim=-1, keepdim=True).clamp(min=1e-6)
    focal_w = (1.0 - p_t).pow(gamma)

    # Weighted CE
    if weight is not None:
        # Broadcast per-class weight across batch
        ce = -(soft_labels * log_probs * weight.unsqueeze(0)).sum(dim=-1)
    else:
        ce = -(soft_labels * log_probs).sum(dim=-1)

    return (focal_w.squeeze(-1) * ce).mean()


__all__ = [
    "AugCfg",
    "DEFAULT_AUG",
    "specaugment",
    "mixup_paired",
    "modality_dropout",
    "ecg_shift_jitter",
    "augment_batch",
    "soft_focal_loss",
]
