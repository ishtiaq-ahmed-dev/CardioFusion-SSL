"""SSL projection heads and reconstruction heads for CardioFusion-SSL.

For cross-modal contrastive learning we need a small MLP projection head on top
of each encoder's pooled embedding. For the auxiliary masked-reconstruction loss
we also need light reconstruction heads that map token embeddings back to the
input space (raw ECG samples and mel-spectrogram patches).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from configs import CFG


# --------------------------------------------------------------------- projection head
class ProjectionHead(nn.Module):
    """2-layer MLP projector used in SimCLR / CLIP. Produces L2-normalised vectors."""

    def __init__(self, in_dim: int = CFG.D_MODEL,
                 hidden_dim: int = CFG.D_MODEL,
                 out_dim: int = CFG.SSL_PROJ_DIM,
                 use_bn: bool = True):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers += [nn.GELU(), nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = nn.functional.normalize(z, dim=-1)
        return z


# --------------------------------------------------------------------- reconstruction heads
class ECGReconHead(nn.Module):
    """Token sequence -> reconstructed ECG patches.

    For masked-signal modelling: predict the original (B, 1, ECG_LEN) signal from
    the encoder's token sequence (B, T, D). We project each token back to a
    `patch`-sized chunk of raw samples and concatenate.
    """

    def __init__(self, d_model: int = CFG.D_MODEL,
                 patch: int = CFG.ECG_PATCH):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(d_model, patch)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, D) -> (B, T, patch) -> (B, 1, T*patch)
        x = self.proj(tokens)                              # (B, T, patch)
        x = x.reshape(x.size(0), 1, -1)
        return x


class PCGReconHead(nn.Module):
    """Token sequence -> reconstructed mel-spectrogram patches.

    Mirrors ECGReconHead; each token decodes a (patch_f x patch_t) mel patch.
    """

    def __init__(self, d_model: int = CFG.D_MODEL,
                 patch_f: int = CFG.PCG_PATCH_F,
                 patch_t: int = CFG.PCG_PATCH_T,
                 mel_n: int = CFG.MEL_N):
        super().__init__()
        self.patch_f = patch_f
        self.patch_t = patch_t
        self.mel_n = mel_n
        self.n_f = mel_n // patch_f
        self.proj = nn.Linear(d_model, patch_f * patch_t)

    def forward(self, tokens: torch.Tensor, mel_t: int) -> torch.Tensor:
        """Args:
            tokens: (B, n_f * n_t, D)  flattened patch tokens
            mel_t:  full mel spectrogram time width (to know n_t at runtime)
        Returns:
            mel_hat: (B, 1, mel_n, mel_t_quantised)
        """
        B, N, D = tokens.shape
        n_t = N // self.n_f
        x = self.proj(tokens)                          # (B, N, patch_f*patch_t)
        x = x.view(B, self.n_f, n_t, self.patch_f, self.patch_t)
        # arrange to (B, 1, F, T)
        x = x.permute(0, 1, 3, 2, 4).contiguous()      # (B, n_f, patch_f, n_t, patch_t)
        x = x.view(B, 1, self.n_f * self.patch_f, n_t * self.patch_t)
        return x


# --------------------------------------------------------------------- pooled embedding utility
def pool_tokens(tokens: torch.Tensor, mask: torch.Tensor | None = None,
                method: str = "mean") -> torch.Tensor:
    """Pool a token sequence (B, T, D) to (B, D).

    - "mean"   global average over T
    - "cls"    take token 0 (assumes a [CLS] token was prepended)
    - "attn"   not implemented here (use a separate head if needed)
    """
    if method == "cls":
        return tokens[:, 0, :]
    # mean (optionally masked)
    if mask is None:
        return tokens.mean(dim=1)
    m = mask.unsqueeze(-1).float()                    # (B, T, 1)
    s = (tokens * m).sum(dim=1)
    n = m.sum(dim=1).clamp(min=1.0)
    return s / n
