"""PCG encoder for CardioFusion-SSL.

Audio Spectrogram Transformer (AST) variant. Takes a log-mel spectrogram of the
PCG signal and returns a token sequence (B, T, D_MODEL) for hierarchical fusion.

Two paths are supported, selected via `CFG.PCG_BACKBONE`:

  - "ast"               Pure Transformer over 2D mel-patches (default).
                        If `CFG.PCG_USE_AUDIOSET_INIT` and HuggingFace
                        transformers is installed, the patch embedding +
                        first few blocks are initialised from AudioSet-
                        pretrained AST weights (MIT/ast-finetuned-audioset).
  - "cnn_transformer"   CNN stem (ResBlock-like) -> Transformer encoder.
                        Smaller, faster, no pretrained init.

Input :  (B, 1, MEL_N, MEL_T)        log-mel spectrogram
Output:  (B, T_pcg, D_MODEL)         token sequence for fusion
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import CFG


# --------------------------------------------------------------------- positional encoding (2D)
class _LearnedPosEmbed(nn.Module):
    """Learned absolute positional embeddings for an N-token sequence."""

    def __init__(self, n_tokens: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) — supports variable N up to self.pe.size(1)
        n = x.size(1)
        return x + self.pe[:, :n]


# --------------------------------------------------------------------- 2D patch embedding
class _MelPatchEmbed(nn.Module):
    """Splits the mel spectrogram into non-overlapping (f x t) patches,
    flattens, linearly projects to d_model.
    """

    def __init__(self, mel_n: int = CFG.MEL_N,
                 patch_f: int = CFG.PCG_PATCH_F,
                 patch_t: int = CFG.PCG_PATCH_T,
                 d_model: int = CFG.D_MODEL):
        super().__init__()
        assert mel_n % patch_f == 0, (
            f"mel_n ({mel_n}) must be divisible by patch_f ({patch_f})"
        )
        self.patch_f = patch_f
        self.patch_t = patch_t
        self.proj = nn.Conv2d(1, d_model,
                              kernel_size=(patch_f, patch_t),
                              stride=(patch_f, patch_t))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, 1, F, T)
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        x = self.proj(mel)                          # (B, D, F', T')
        B, D, Fp, Tp = x.shape
        x = x.flatten(2).transpose(1, 2)            # (B, F'*T', D)
        return self.norm(x), (Fp, Tp)


# --------------------------------------------------------------------- transformer block
class _ASTBlock(nn.Module):
    def __init__(self, d_model: int = CFG.D_MODEL,
                 n_heads: int = CFG.N_HEADS,
                 dropout: float = CFG.DROPOUT,
                 ffn_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
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


# --------------------------------------------------------------------- CNN-Transformer alternative
class _ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True):
        super().__init__()
        s = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if downsample or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=s, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.short = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(h + self.short(x))


class _CNNStem(nn.Module):
    """Downsampling CNN stem reducing (F, T) before tokenisation."""

    def __init__(self, d_model: int = CFG.D_MODEL):
        super().__init__()
        c = (32, 64, 128, d_model)
        self.stem = nn.Sequential(
            _ResBlock2D(1, c[0]),
            _ResBlock2D(c[0], c[1]),
            _ResBlock2D(c[1], c[2]),
            _ResBlock2D(c[2], c[3], downsample=False),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        return self.stem(mel)   # (B, D, F/16, T/16)


# --------------------------------------------------------------------- encoder
class PCGEncoder(nn.Module):
    """Log-mel PCG -> token sequence (B, T, D_MODEL)."""

    def __init__(self,
                 d_model: int = CFG.D_MODEL,
                 n_blocks: int = CFG.PCG_DEPTH,
                 backbone: Optional[str] = None,
                 dropout: float = CFG.DROPOUT,
                 mel_n: int = CFG.MEL_N,
                 mel_t: Optional[int] = None):
        super().__init__()
        backbone = backbone or CFG.PCG_BACKBONE
        self.backbone = backbone

        # estimate mel_t at construction time
        if mel_t is None:
            mel_t = 1 + (CFG.PCG_LEN - CFG.MEL_WIN) // CFG.MEL_HOP + 1

        if backbone == "ast":
            self.embed = _MelPatchEmbed(mel_n=mel_n, d_model=d_model,
                                        patch_f=CFG.PCG_PATCH_F,
                                        patch_t=CFG.PCG_PATCH_T)
            n_f = mel_n // CFG.PCG_PATCH_F
            n_t = max(1, mel_t // CFG.PCG_PATCH_T)
            self.pos = _LearnedPosEmbed(n_f * n_t + 32, d_model)  # +slack for variable T
        elif backbone == "cnn_transformer":
            self.cnn = _CNNStem(d_model=d_model)
            # CNN halves F and T 3 times -> /8; estimate token count
            n_tokens = (mel_n // 8) * (mel_t // 8)
            self.pos = _LearnedPosEmbed(n_tokens + 32, d_model)
        else:
            raise ValueError(f"Unknown PCG backbone: {backbone}")

        self.blocks = nn.ModuleList([
            _ASTBlock(d_model=d_model, dropout=dropout) for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Args:
            mel: (B, 1, F, T) or (B, F, T) log-mel spectrogram
        Returns:
            (B, T_pcg, D_MODEL)
        """
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)

        if self.backbone == "ast":
            x, _grid = self.embed(mel)
        else:
            feat = self.cnn(mel)                 # (B, D, F', T')
            B, D, Fp, Tp = feat.shape
            x = feat.flatten(2).transpose(1, 2)  # (B, F'*T', D)

        x = self.pos(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
