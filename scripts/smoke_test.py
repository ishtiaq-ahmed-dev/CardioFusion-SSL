"""Smoke test for CardioFusion-SSL.

Builds the full model, runs forward + backward on a synthetic batch in both
supervised and SSL modes, and prints parameter counts and output shapes. No
real data, no GPU required (runs on CPU in <30 s).

Usage (from project root):
    python -m scripts.smoke_test
"""
from __future__ import annotations

import sys
from pathlib import Path

# allow running as a script from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from configs import CFG
from models import CardioFusionSSL
from pretraining.contrastive import CrossModalContrastiveLoss, info_nce_accuracy


def _expected_mel_t() -> int:
    return 1 + (CFG.PCG_LEN - CFG.MEL_WIN) // CFG.MEL_HOP + 1


def synthetic_batch(B: int = 4, device: str = "cpu") -> dict:
    mel_t = _expected_mel_t()
    return {
        "ecg":     torch.randn(B, 1, CFG.ECG_LEN, device=device),
        "pcg_mel": torch.randn(B, 1, CFG.MEL_N, mel_t, device=device),
        "has_ecg": torch.ones(B, device=device),
        "has_pcg": torch.ones(B, device=device),
        "label":   torch.randint(0, CFG.N_BINARY, (B,), device=device),
        "ecg_target": torch.randn(B, 1, CFG.ECG_LEN, device=device),
        "mel_target": torch.randn(B, 1, CFG.MEL_N, mel_t, device=device),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device = {device}")
    print(f"[smoke] expected mel_t = {_expected_mel_t()}")

    torch.manual_seed(CFG.SEED)
    model = CardioFusionSSL(enable_ssl=True, enable_recon=True).to(device)

    # ----- parameter counts
    counts = model.count_parameters()
    print("\n[smoke] parameter counts:")
    for k, v in counts.items():
        print(f"  {k:<15s} {v:>12,d}")

    # ----- supervised forward
    batch = synthetic_batch(B=4, device=device)
    out_sup = model(batch, mode="supervised")
    assert out_sup["logits"].shape == (4, CFG.N_BINARY), out_sup["logits"].shape
    assert out_sup["features"].shape == (4, model.fusion.out_dim), out_sup["features"].shape
    print(f"\n[smoke] supervised OK  logits {tuple(out_sup['logits'].shape)} "
          f" features {tuple(out_sup['features'].shape)}")

    # ----- supervised backward
    loss = F.cross_entropy(out_sup["logits"], batch["label"])
    loss.backward()
    print(f"[smoke] supervised backward OK   loss = {loss.item():.4f}")
    model.zero_grad(set_to_none=True)

    # ----- SSL forward
    out_ssl = model(batch, mode="ssl")
    assert out_ssl["z_ecg"].shape == (4, CFG.SSL_PROJ_DIM)
    assert out_ssl["z_pcg"].shape == (4, CFG.SSL_PROJ_DIM)
    print(f"[smoke] ssl forward OK   z_ecg {tuple(out_ssl['z_ecg'].shape)} "
          f" z_pcg {tuple(out_ssl['z_pcg'].shape)}")

    if "ecg_recon" in out_ssl:
        # Recon length may differ from ECG_LEN by < ECG_PATCH due to patching
        assert out_ssl["ecg_recon"].shape[0] == 4
        print(f"[smoke] ssl ecg_recon shape {tuple(out_ssl['ecg_recon'].shape)}")
    if "mel_recon" in out_ssl:
        assert out_ssl["mel_recon"].shape[0] == 4
        print(f"[smoke] ssl mel_recon shape {tuple(out_ssl['mel_recon'].shape)}")

    # ----- contrastive loss
    crit = CrossModalContrastiveLoss()
    res = crit(out_ssl["z_ecg"], out_ssl["z_pcg"])
    print(f"[smoke] contrastive loss = {res['loss'].item():.4f}  "
          f"(cross = {res['loss_cross'].item():.4f})")
    print(f"[smoke] retrieval accuracy: {res['acc']}")

    res["loss"].backward()
    print("[smoke] ssl backward OK")

    # ----- missing-modality sanity (ECG-only sample)
    batch2 = synthetic_batch(B=4, device=device)
    batch2["has_pcg"] = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device)
    batch2["pcg_mel"] = torch.zeros_like(batch2["pcg_mel"])   # zeroed: should be replaced by token
    out_missing = model(batch2, mode="supervised")
    assert out_missing["logits"].shape == (4, CFG.N_BINARY)
    print(f"[smoke] missing-modality forward OK  "
          f"logits {tuple(out_missing['logits'].shape)}")

    print("\n[smoke] ALL TESTS PASSED")


if __name__ == "__main__":
    main()
