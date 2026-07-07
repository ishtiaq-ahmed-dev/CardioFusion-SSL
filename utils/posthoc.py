"""Post-hoc probabilistic tools that improve a trained ensemble without retraining.

  - temperature_scaling  : one-scalar calibration on validation logits
  - optimal_ensemble     : weighted soft-vote with weights learned per-model
                           on out-of-fold validation predictions

Both operate on already-saved per-fold predictions from the v2 fine-tune run.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
#  Temperature scaling (Guo et al. 2017)
# ═══════════════════════════════════════════════════════════════════════════
def _prob_to_logits(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Invert softmax on binary probabilities into logits (2-class)."""
    p = np.clip(p, eps, 1 - eps)
    # Two-class logit representation: [0, log(p / (1-p))]
    return np.stack([np.zeros_like(p), np.log(p / (1 - p))], axis=-1)


def fit_temperature(y_prob_val: np.ndarray, y_true_val: np.ndarray,
                    n_iter: int = 200) -> float:
    """Fit a single temperature T that minimises NLL on the validation set.

    y_prob_val : (N,) predicted positive-class probabilities (post-softmax)
    y_true_val : (N,) binary labels {0, 1}

    Returns the scalar T. Apply as: p_calibrated = sigmoid(logit(p) / T)
    """
    logits = torch.from_numpy(_prob_to_logits(y_prob_val)).float()
    y = torch.from_numpy(y_true_val.astype(np.int64))
    T = torch.nn.Parameter(torch.ones(1))

    def nll():
        return F.cross_entropy(logits / T.clamp(min=0.05), y)

    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=n_iter)
    def closure():
        opt.zero_grad(); loss = nll(); loss.backward(); return loss
    opt.step(closure)
    return float(T.detach().clamp(min=0.05).item())


def apply_temperature(y_prob: np.ndarray, T: float, eps: float = 1e-6) -> np.ndarray:
    """Divide the logit by T and take sigmoid."""
    p = np.clip(y_prob, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-logit / max(T, 0.05)))


# ═══════════════════════════════════════════════════════════════════════════
#  Optimal ensemble weights
# ═══════════════════════════════════════════════════════════════════════════
def fit_ensemble_weights(prob_matrix: np.ndarray, y_true: np.ndarray,
                         reg: float = 1e-3, n_iter: int = 500,
                         lr: float = 0.05) -> np.ndarray:
    """Learn convex weights on model logits that minimise validation NLL.

    prob_matrix : (M, N) predicted probabilities from M ensemble members
    y_true      : (N,) binary labels

    Returns a length-M weight vector with sum=1, entries in [0, 1].
    """
    M, N = prob_matrix.shape
    logits = torch.from_numpy(np.log(np.clip(prob_matrix, 1e-6, 1 - 1e-6)
                                     / np.clip(1 - prob_matrix, 1e-6, 1 - 1e-6))).float()
    y = torch.from_numpy(y_true.astype(np.float32))

    # Parameterise weights via softmax over an unconstrained vector
    alpha = torch.nn.Parameter(torch.zeros(M))
    opt = torch.optim.Adam([alpha], lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()
        w = F.softmax(alpha, dim=0).unsqueeze(1)         # (M, 1)
        avg_logit = (w * logits).sum(dim=0)              # (N,)
        p = torch.sigmoid(avg_logit)
        nll = F.binary_cross_entropy(p, y)
        # Small entropy regulariser to prevent collapse to one model
        entropy = -(w.squeeze(1) * torch.log(w.squeeze(1) + 1e-9)).sum()
        loss = nll - reg * entropy
        loss.backward()
        opt.step()

    return F.softmax(alpha.detach(), dim=0).numpy()


def apply_ensemble_weights(prob_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted average of model probabilities in logit space."""
    logits = np.log(np.clip(prob_matrix, 1e-6, 1 - 1e-6)
                    / np.clip(1 - prob_matrix, 1e-6, 1 - 1e-6))
    avg = (weights[:, None] * logits).sum(axis=0)
    return 1.0 / (1.0 + np.exp(-avg))


__all__ = [
    "fit_temperature", "apply_temperature",
    "fit_ensemble_weights", "apply_ensemble_weights",
]
