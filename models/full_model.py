"""End-to-end CardioFusion-SSL model composition.

Glues ECG encoder + PCG encoder + hierarchical fusion + classifier head.
Exposes a single forward that handles paired and unpaired samples uniformly
(via the missing-modality tokens inside the fusion module).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from configs import CFG
from models.mamba_ecg import ECGEncoder
from models.ast_pcg import PCGEncoder
from models.hier_fusion import HierFusion
from models.ssl_heads import ProjectionHead, ECGReconHead, PCGReconHead, pool_tokens


# --------------------------------------------------------------------- classifier
class _ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = CFG.N_BINARY,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------- main model
class CardioFusionSSL(nn.Module):
    """Full pipeline: encoders -> fusion -> heads.

    Forward modes are controlled by `mode`:
      - "supervised"   returns classifier logits
      - "ssl"          returns SSL projection embeddings + (optional) reconstructions

    Batch fields:
      ecg     : (B, 1, ECG_LEN)            raw ECG  (zeros allowed if has_ecg=0)
      pcg_mel : (B, 1, MEL_N, MEL_T)       log-mel  (zeros allowed if has_pcg=0)
      has_ecg : (B,) float                 1.0 if real ECG present
      has_pcg : (B,) float                 1.0 if real PCG present
    """

    def __init__(self, n_classes: int = CFG.N_BINARY,
                 enable_ssl: bool = True,
                 enable_recon: bool = True):
        super().__init__()
        self.ecg_enc = ECGEncoder()
        self.pcg_enc = PCGEncoder()
        self.fusion = HierFusion()
        self.classifier = _ClassifierHead(in_dim=self.fusion.out_dim, n_classes=n_classes)

        if enable_ssl:
            self.proj_ecg = ProjectionHead(in_dim=CFG.D_MODEL)
            self.proj_pcg = ProjectionHead(in_dim=CFG.D_MODEL)
        else:
            self.proj_ecg = None
            self.proj_pcg = None

        if enable_recon:
            self.recon_ecg = ECGReconHead()
            self.recon_pcg = PCGReconHead()
        else:
            self.recon_ecg = None
            self.recon_pcg = None

    # ---------------- forward helpers
    def encode_ecg(self, ecg: torch.Tensor) -> torch.Tensor:
        return self.ecg_enc(ecg)

    def encode_pcg(self, mel: torch.Tensor) -> torch.Tensor:
        return self.pcg_enc(mel)

    # ---------------- main forward
    def forward(self, batch: Dict[str, torch.Tensor],
                mode: str = "supervised") -> Dict[str, torch.Tensor]:
        ecg_tok = self.encode_ecg(batch["ecg"])               # (B, T_e, D)
        pcg_tok = self.encode_pcg(batch["pcg_mel"])           # (B, T_p, D)

        if mode == "ssl":
            return self._forward_ssl(ecg_tok, pcg_tok, batch)
        else:
            return self._forward_supervised(ecg_tok, pcg_tok, batch)

    def _forward_supervised(self,
                            ecg_tok: torch.Tensor,
                            pcg_tok: torch.Tensor,
                            batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        has_ecg = batch.get("has_ecg")
        has_pcg = batch.get("has_pcg")
        fused = self.fusion(ecg_tok, pcg_tok, has_ecg=has_ecg, has_pcg=has_pcg)
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "features": fused,
            "attn": getattr(self.fusion, "last_attn_per_scale", None),
        }

    def _forward_ssl(self,
                     ecg_tok: torch.Tensor,
                     pcg_tok: torch.Tensor,
                     batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.proj_ecg is None or self.proj_pcg is None:
            raise RuntimeError("SSL projection heads disabled; pass enable_ssl=True at construction")

        e_pool = pool_tokens(ecg_tok)                          # (B, D)
        p_pool = pool_tokens(pcg_tok)
        z_e = self.proj_ecg(e_pool)
        z_p = self.proj_pcg(p_pool)

        out = {"z_ecg": z_e, "z_pcg": z_p,
               "tok_ecg": ecg_tok, "tok_pcg": pcg_tok}

        if self.recon_ecg is not None and "ecg_target" in batch:
            out["ecg_recon"] = self.recon_ecg(ecg_tok)
        if self.recon_pcg is not None and "mel_target" in batch:
            mel_t = batch["mel_target"].size(-1)
            out["mel_recon"] = self.recon_pcg(pcg_tok, mel_t=mel_t)

        return out

    # ---------------- IO helpers
    def freeze_encoders(self) -> None:
        for p in self.ecg_enc.parameters():
            p.requires_grad_(False)
        for p in self.pcg_enc.parameters():
            p.requires_grad_(False)

    def unfreeze_encoders(self) -> None:
        for p in self.ecg_enc.parameters():
            p.requires_grad_(True)
        for p in self.pcg_enc.parameters():
            p.requires_grad_(True)

    def count_parameters(self) -> dict:
        def c(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "ecg_encoder": c(self.ecg_enc),
            "pcg_encoder": c(self.pcg_enc),
            "fusion":      c(self.fusion),
            "classifier":  c(self.classifier),
            "proj_ecg":    c(self.proj_ecg) if self.proj_ecg is not None else 0,
            "proj_pcg":    c(self.proj_pcg) if self.proj_pcg is not None else 0,
            "total":       c(self),
        }
