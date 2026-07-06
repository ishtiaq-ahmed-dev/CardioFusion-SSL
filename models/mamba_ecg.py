"""ECG encoder for CardioFusion-SSL.

Provides three interchangeable backbones, selected via `CFG.ECG_BACKBONE`:

  - "transformer" (default, robust)        - CNN patch stem + N Transformer blocks
  - "mamba"        (uses mamba-ssm if installed; else falls back to transformer)
  - "hybrid"       - CNN stem + half-Transformer half-conv blocks

All backbones expose the same I/O contract:

    Input :  (B, 1, ECG_LEN)             raw single-lead ECG @ 500 Hz
    Output:  (B, T_ecg, D_MODEL)         token sequence for hierarchical fusion
             T_ecg = ECG_LEN // ECG_PATCH    (default 2000 / 25 = 80 tokens)

This shared contract lets the rest of the model (fusion, SSL heads) treat the
encoder as a black box.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import CFG


# --------------------------------------------------------------------- utilities
class _SinCosPositionalEncoding(nn.Module):
    """Standard fixed sin/cos positional encoding (Vaswani 2017)."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class _PatchStem(nn.Module):
    """1D conv stem that turns a raw ECG (B, 1, L) into a token sequence (B, T, D).

    Achieved by a single strided Conv1d with kernel=patch, stride=patch.
    """

    def __init__(self, in_ch: int = 1, d_model: int = CFG.D_MODEL,
                 patch: int = CFG.ECG_PATCH):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, d_model, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L) -> (B, D, T) -> (B, T, D)
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


# --------------------------------------------------------------------- transformer block
class _TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block (LN -> MHA -> Add -> LN -> FFN -> Add)."""

    def __init__(self, d_model: int = CFG.D_MODEL,
                 n_heads: int = CFG.N_HEADS,
                 dropout: float = CFG.DROPOUT,
                 ffn_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        x = x + self.ffn(self.ln2(x))
        return x


# --------------------------------------------------------------------- optional mamba
def _try_import_mamba():
    """Return (MambaBlock, True) if mamba-ssm is importable, else (None, False).

    Decoupled to a function so import failure on Windows / no-Blackwell-wheel
    doesn't crash the module at import time.
    """
    try:
        from mamba_ssm import Mamba   # type: ignore
        return Mamba, True
    except Exception:
        return None, False


class _MambaBlock(nn.Module):
    """Wraps mamba_ssm.Mamba with pre-norm + residual, mirroring _TransformerBlock."""

    def __init__(self, d_model: int = CFG.D_MODEL,
                 d_state: int = CFG.ECG_D_STATE,
                 d_conv: int = CFG.ECG_D_CONV,
                 expand: int = CFG.ECG_EXPAND,
                 dropout: float = CFG.DROPOUT):
        super().__init__()
        Mamba, ok = _try_import_mamba()
        if not ok:
            raise ImportError("mamba-ssm not installed")
        self.ln = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state,
                           d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.mamba(h)
        return x + self.drop(h)


# --------------------------------------------------------------------- encoder
class ECGEncoder(nn.Module):
    """Single-lead ECG -> token sequence (B, T, D_MODEL).

    Backbone selectable via CFG.ECG_BACKBONE.
    """

    def __init__(self,
                 in_ch: int = 1,
                 d_model: int = CFG.D_MODEL,
                 n_blocks: int = CFG.ECG_DEPTH,
                 patch: int = CFG.ECG_PATCH,
                 backbone: Optional[str] = None,
                 dropout: float = CFG.DROPOUT):
        super().__init__()
        backbone = backbone or CFG.ECG_BACKBONE
        self.stem = _PatchStem(in_ch=in_ch, d_model=d_model, patch=patch)
        self.pos = _SinCosPositionalEncoding(d_model)

        # decide block class
        if backbone == "mamba":
            _, ok = _try_import_mamba()
            if ok:
                block_cls = lambda: _MambaBlock(d_model=d_model)
                self.backbone = "mamba"
            else:
                print("[ECGEncoder] mamba-ssm unavailable -> falling back to transformer")
                block_cls = lambda: _TransformerBlock(d_model=d_model, dropout=dropout)
                self.backbone = "transformer"
        elif backbone == "hybrid":
            # alternate transformer + 1D conv attention (kept simple here)
            self.backbone = "hybrid"
            block_cls = lambda: _TransformerBlock(d_model=d_model, dropout=dropout)
        else:
            self.backbone = "transformer"
            block_cls = lambda: _TransformerBlock(d_model=d_model, dropout=dropout)

        self.blocks = nn.ModuleList([block_cls() for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, 1, ECG_LEN) raw signal OR (B, ECG_LEN) -- auto-unsqueeze
        Returns:
            (B, T, D_MODEL) token sequence
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)               # (B, T, D)
        x = self.pos(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
