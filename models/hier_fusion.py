"""Hierarchical multi-scale cross-modal fusion.

Takes ECG and PCG token sequences from the two encoders and fuses them at
multiple temporal resolutions, then aggregates a fixed-size fused embedding for
downstream heads.

Why hierarchical
----------------
ECG and PCG are coupled at *several* time scales physiologically:
  - sub-cycle:    S1 follows the R-peak by ~30-50 ms; S2 follows the T-wave.
  - cycle:        each heartbeat is a self-contained electromechanical event.
  - recording:    rhythm trends, murmur persistence are global.

Prior fusion work uses one global cross-attention and loses the per-scale
information. We do bidirectional cross-attention separately at coarse / medium /
fine scales (token-grouping factors from CFG.FUSION_SCALES) and concatenate.

Missing modality
----------------
If a sample provides only one modality, the absent modality is replaced by a
learned token broadcast to the appropriate scale before cross-attention. This
preserves the architecture across paired, ECG-only and PCG-only samples.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import CFG


# --------------------------------------------------------------------- helpers
def _resample_tokens(x: torch.Tensor, n_target: int) -> torch.Tensor:
    """Adaptive pool a token sequence (B, T, D) to (B, n_target, D)."""
    B, T, D = x.shape
    if T == n_target:
        return x
    # (B, T, D) -> (B, D, T) -> pool -> (B, D, n_target) -> (B, n_target, D)
    x = x.transpose(1, 2)
    x = F.adaptive_avg_pool1d(x, n_target)
    return x.transpose(1, 2)


# --------------------------------------------------------------------- cross-attention block
class _BiCrossAttnBlock(nn.Module):
    """One block of bidirectional cross-attention with pre-norm + FFN."""

    def __init__(self, d_model: int = CFG.D_MODEL,
                 n_heads: int = CFG.N_HEADS,
                 dropout: float = CFG.FUSION_DROPOUT,
                 ffn_mult: int = 4):
        super().__init__()
        self.ln_e1 = nn.LayerNorm(d_model)
        self.ln_p1 = nn.LayerNorm(d_model)
        self.attn_e = nn.MultiheadAttention(d_model, n_heads,
                                            dropout=dropout, batch_first=True)
        self.attn_p = nn.MultiheadAttention(d_model, n_heads,
                                            dropout=dropout, batch_first=True)
        self.ln_e2 = nn.LayerNorm(d_model)
        self.ln_p2 = nn.LayerNorm(d_model)
        self.ffn_e = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_p = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)
        self.last_attn = None

    def forward(self, e: torch.Tensor, p: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ECG queries PCG
        he = self.ln_e1(e)
        hp = self.ln_p1(p)
        a_e, w_e = self.attn_e(he, hp, hp, need_weights=True, average_attn_weights=True)
        e = e + self.drop(a_e)
        # PCG queries ECG (uses updated ECG)
        hp = self.ln_p1(p)            # reuse pre-norm input for symmetry
        he = self.ln_e1(e)
        a_p, w_p = self.attn_p(hp, he, he, need_weights=True, average_attn_weights=True)
        p = p + self.drop(a_p)
        # FFNs
        e = e + self.ffn_e(self.ln_e2(e))
        p = p + self.ffn_p(self.ln_p2(p))
        self.last_attn = {"e_attends_p": w_e.detach(), "p_attends_e": w_p.detach()}
        return e, p


# --------------------------------------------------------------------- scale-specific fusion stack
class _ScaleFusion(nn.Module):
    """Run FUSION_DEPTH bidirectional cross-attention blocks at a single scale."""

    def __init__(self, depth: int = CFG.FUSION_DEPTH,
                 d_model: int = CFG.D_MODEL):
        super().__init__()
        self.blocks = nn.ModuleList([
            _BiCrossAttnBlock(d_model=d_model) for _ in range(depth)
        ])
        self.ln_e = nn.LayerNorm(d_model)
        self.ln_p = nn.LayerNorm(d_model)

    def forward(self, e: torch.Tensor, p: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        for blk in self.blocks:
            e, p = blk(e, p)
        return self.ln_e(e), self.ln_p(p)


# --------------------------------------------------------------------- missing modality tokens
class _MissingTokens(nn.Module):
    def __init__(self, d_model: int = CFG.D_MODEL):
        super().__init__()
        self.ecg_missing = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pcg_missing = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.ecg_missing, std=0.02)
        nn.init.trunc_normal_(self.pcg_missing, std=0.02)

    def fill(self, x: torch.Tensor, has: torch.Tensor, which: str) -> torch.Tensor:
        """Replace whole-sequence with learned token where has==0."""
        token = self.ecg_missing if which == "ecg" else self.pcg_missing
        if has is None:
            return x
        # broadcast token to (B, T, D) for samples with has=0
        mask = (has < 0.5).view(-1, 1, 1)             # (B, 1, 1)
        token_expanded = token.expand_as(x)            # (B, T, D)
        return torch.where(mask, token_expanded, x)


# --------------------------------------------------------------------- main fusion module
class HierFusion(nn.Module):
    """Hierarchical multi-scale ECG+PCG fusion.

    Inputs
    ------
    ecg_tokens : (B, T_e, D)
    pcg_tokens : (B, T_p, D)
    has_ecg    : (B,)  optional, 1.0 if real ECG present else 0.0
    has_pcg    : (B,)  optional, 1.0 if real PCG present else 0.0

    Output
    ------
    fused_embedding : (B, 2 * D * len(FUSION_SCALES))
    """

    def __init__(self,
                 d_model: int = CFG.D_MODEL,
                 scales: Tuple[int, ...] = CFG.FUSION_SCALES,
                 depth: int = CFG.FUSION_DEPTH):
        super().__init__()
        self.scales = scales
        self.d_model = d_model
        self.tokens = _MissingTokens(d_model=d_model)
        self.per_scale = nn.ModuleDict({
            f"s{s}": _ScaleFusion(depth=depth, d_model=d_model) for s in scales
        })
        self.out_dim = 2 * d_model * len(scales)

    def forward(self,
                ecg_tokens: torch.Tensor,
                pcg_tokens: torch.Tensor,
                has_ecg: Optional[torch.Tensor] = None,
                has_pcg: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. handle missing modality (replace whole sequence with learned token)
        ecg_tokens = self.tokens.fill(ecg_tokens, has_ecg, "ecg")
        pcg_tokens = self.tokens.fill(pcg_tokens, has_pcg, "pcg")

        scale_embeds = []
        self.last_attn_per_scale = {}
        for s in self.scales:
            e_s = _resample_tokens(ecg_tokens, s)
            p_s = _resample_tokens(pcg_tokens, s)
            e_s, p_s = self.per_scale[f"s{s}"](e_s, p_s)
            # store attention from last block for explainability
            self.last_attn_per_scale[s] = self.per_scale[f"s{s}"].blocks[-1].last_attn
            # mean-pool each modality at this scale
            scale_embeds.append(torch.cat([e_s.mean(dim=1), p_s.mean(dim=1)], dim=-1))
        return torch.cat(scale_embeds, dim=-1)         # (B, 2 * D * n_scales)
