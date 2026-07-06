"""Cross-modal contrastive learning losses (InfoNCE / NT-Xent).

Trains ECG and PCG encoders so that paired (same-clip) embeddings are close in
projection space and mismatched pairs are far. Same machinery powers within-
modality SimCLR-style augment contrast as a regulariser.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import CFG


# --------------------------------------------------------------------- core InfoNCE
def info_nce_loss(z_a: torch.Tensor,
                  z_b: torch.Tensor,
                  temperature: float = CFG.SSL_TEMPERATURE,
                  symmetric: bool = True) -> torch.Tensor:
    """Symmetric InfoNCE / NT-Xent loss between two L2-normalised batches.

    z_a, z_b : (B, d) L2-normalised projection embeddings (same indexing -> positives)

    Returns a scalar loss = 0.5 * (CE(sim, identity) + CE(sim^T, identity)) when
    `symmetric=True`, else just CE(sim, identity).
    """
    if z_a.size(0) != z_b.size(0):
        raise ValueError(f"batch mismatch: {z_a.shape} vs {z_b.shape}")
    B = z_a.size(0)
    if B < 2:
        # InfoNCE is undefined with a single positive and no negatives.
        return torch.zeros((), device=z_a.device, dtype=z_a.dtype)

    sim = z_a @ z_b.T / temperature                    # (B, B)
    target = torch.arange(B, device=sim.device)
    loss_ab = F.cross_entropy(sim, target)
    if not symmetric:
        return loss_ab
    loss_ba = F.cross_entropy(sim.T, target)
    return 0.5 * (loss_ab + loss_ba)


# --------------------------------------------------------------------- accuracy diagnostic
@torch.no_grad()
def info_nce_accuracy(z_a: torch.Tensor, z_b: torch.Tensor) -> dict:
    """Compute top-1 / top-5 retrieval accuracy on the in-batch matrix.

    Useful for monitoring SSL progress; expect top-1 to climb from 1/B (random)
    toward 1.0 as the encoders learn cross-modal alignment.
    """
    B = z_a.size(0)
    if B < 2:
        return {"top1_ab": 0.0, "top5_ab": 0.0, "top1_ba": 0.0, "top5_ba": 0.0}
    sim = z_a @ z_b.T
    target = torch.arange(B, device=sim.device)
    k5 = min(5, B)

    top1_ab = (sim.argmax(dim=1) == target).float().mean().item()
    top5_ab = (sim.topk(k5, dim=1).indices.eq(target.unsqueeze(1)).any(dim=1)
               .float().mean().item())
    top1_ba = (sim.argmax(dim=0) == target).float().mean().item()
    top5_ba = (sim.topk(k5, dim=0).indices.eq(target.unsqueeze(0)).any(dim=0)
               .float().mean().item())
    return {"top1_ab": top1_ab, "top5_ab": top5_ab,
            "top1_ba": top1_ba, "top5_ba": top5_ba}


# --------------------------------------------------------------------- wrapped loss with optional masked targets
class CrossModalContrastiveLoss(nn.Module):
    """Bundles InfoNCE for cross-modal + optional within-modality augment contrast.

    The two SSL streams come from two augmentations of each sample:
        - the cross-modal positive: ECG view 1 vs PCG view 1
        - (optional) within-modality positives: ECG view 1 vs ECG view 2

    The within-modality contrast term acts as an invariance regulariser similar
    to SimCLR; weight is `CFG.SSL_LOSS_W_MOD_CONTRAST`.
    """

    def __init__(self,
                 temperature: float = CFG.SSL_TEMPERATURE,
                 w_cross: float = CFG.SSL_LOSS_W_CONTRAST,
                 w_mod: float = CFG.SSL_LOSS_W_MOD_CONTRAST):
        super().__init__()
        self.temperature = temperature
        self.w_cross = w_cross
        self.w_mod = w_mod

    def forward(self,
                z_ecg_v1: torch.Tensor,
                z_pcg_v1: torch.Tensor,
                z_ecg_v2: Optional[torch.Tensor] = None,
                z_pcg_v2: Optional[torch.Tensor] = None) -> dict:
        out = {}
        loss_cross = info_nce_loss(z_ecg_v1, z_pcg_v1, self.temperature)
        out["loss_cross"] = loss_cross
        total = self.w_cross * loss_cross

        if (z_ecg_v2 is not None) and (self.w_mod > 0):
            loss_ecg_aug = info_nce_loss(z_ecg_v1, z_ecg_v2, self.temperature)
            total = total + self.w_mod * loss_ecg_aug
            out["loss_ecg_aug"] = loss_ecg_aug
        if (z_pcg_v2 is not None) and (self.w_mod > 0):
            loss_pcg_aug = info_nce_loss(z_pcg_v1, z_pcg_v2, self.temperature)
            total = total + self.w_mod * loss_pcg_aug
            out["loss_pcg_aug"] = loss_pcg_aug

        out["loss"] = total
        out["acc"] = info_nce_accuracy(z_ecg_v1, z_pcg_v1)
        return out
