"""CardioFusion-SSL — single source of truth for all hyperparameters.

Every other module imports CFG from here. No magic numbers anywhere else.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class _CFG:
    # ------------------------------------------------------------------ paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    OLD_PROJECT_ROOT: Path = Path("D:/AI_LAB_RP/CardioSense-AI")
    DATASETS_ROOT: Path = Path("D:/AI_LAB_RP/DATASETS")
    EXTRACTED_ROOT: Path = Path("D:/AI_LAB_RP/DATASETS/_extracted")

    CACHE_DIR: Path = field(init=False)
    CHECKPOINT_DIR: Path = field(init=False)
    RESULTS_DIR: Path = field(init=False)
    LOG_DIR: Path = field(init=False)

    def __post_init__(self):
        self.CACHE_DIR = self.PROJECT_ROOT / "cache"
        self.CHECKPOINT_DIR = self.PROJECT_ROOT / "checkpoints"
        self.RESULTS_DIR = self.PROJECT_ROOT / "results"
        self.LOG_DIR = self.PROJECT_ROOT / "logs"
        for p in (self.CACHE_DIR, self.CHECKPOINT_DIR, self.RESULTS_DIR, self.LOG_DIR):
            os.makedirs(p, exist_ok=True)

    # ------------------------------------------------------------------ reproducibility
    SEED: int = 1337

    # ------------------------------------------------------------------ signal params
    ECG_FS: int = 500
    PCG_FS: int = 2000
    SEG_SECONDS: float = 4.0
    SEG_OVERLAP: float = 0.50
    ECG_LEN: int = 2000           # 500 Hz * 4 s
    PCG_LEN: int = 8000           # 2000 Hz * 4 s

    # ECG filter
    ECG_BAND: Tuple[float, float] = (0.5, 40.0)
    ECG_FILT_ORDER: int = 4
    ECG_NOTCH_HZ: float = 50.0

    # PCG filter
    PCG_BAND: Tuple[float, float] = (25.0, 400.0)
    PCG_FILT_ORDER: int = 4

    # Mel-spectrogram
    MEL_N: int = 128
    MEL_N_FFT: int = 1024
    MEL_HOP: int = 64
    MEL_WIN: int = 1024
    MEL_FMIN: float = 20.0
    MEL_FMAX: float = 600.0
    # MEL_T = 1 + (PCG_LEN - MEL_WIN) // MEL_HOP + 1 -> roughly 110-111 frames

    # ------------------------------------------------------------------ encoder dims
    D_MODEL: int = 256
    N_HEADS: int = 8
    DROPOUT: float = 0.1

    # ECG encoder
    ECG_BACKBONE: str = "mamba"   # "mamba" | "transformer" | "hybrid"
    ECG_DEPTH: int = 6            # number of Mamba/Transformer blocks
    ECG_PATCH: int = 25           # samples per token (500 Hz * 0.05 s = 25)
    ECG_D_STATE: int = 16         # Mamba SSM hidden state
    ECG_D_CONV: int = 4           # Mamba 1D conv kernel
    ECG_EXPAND: int = 2           # Mamba block expansion factor

    # PCG encoder
    PCG_BACKBONE: str = "ast"     # "ast" | "transformer" | "cnn_transformer"
    PCG_PATCH_F: int = 16         # mel-bins per patch
    PCG_PATCH_T: int = 16         # time-frames per patch
    PCG_DEPTH: int = 6
    PCG_USE_AUDIOSET_INIT: bool = True   # if HF AST weights available, init from them

    # ------------------------------------------------------------------ fusion
    FUSION_SCALES: Tuple[int, ...] = (4, 16, 64)   # token-grouping factors -> 3 resolutions
    FUSION_DEPTH: int = 2          # cross-attention blocks per scale
    FUSION_DROPOUT: float = 0.2

    # ------------------------------------------------------------------ SSL
    SSL_PROJ_DIM: int = 128        # projection-head output dim for contrastive
    SSL_TEMPERATURE: float = 0.07
    SSL_MASK_RATIO: float = 0.40   # masked-reconstruction mask fraction
    SSL_LOSS_W_CONTRAST: float = 1.0
    SSL_LOSS_W_RECON: float = 0.5
    SSL_LOSS_W_MOD_CONTRAST: float = 0.5   # within-modality SimCLR-style augment contrastive

    # ------------------------------------------------------------------ class definitions
    BINARY_CLASSES: Tuple[str, ...] = ("Normal", "Abnormal")
    N_BINARY: int = 2

    # ------------------------------------------------------------------ splits
    SUBJECT_DISJOINT: bool = True
    N_FOLDS: int = 10              # bumped from 5 for tighter CI
    VAL_FRAC: float = 0.15         # within each fold's train set

    # ------------------------------------------------------------------ loss
    FOCAL_GAMMA: float = 2.0
    LABEL_SMOOTHING: float = 0.05

    # ------------------------------------------------------------------ training (supervised)
    BATCH_SIZE: int = 32           # paired data is small (~478 records)
    BATCH_SIZE_LARGE: int = 128    # for SSL (unpaired pool is large)
    NUM_WORKERS: int = 0 if os.name == "nt" else 8
    PIN_MEMORY: bool = True
    PREFETCH_FACTOR: int = 4
    PERSISTENT_WORKERS: bool = True

    LR: float = 3e-4
    LR_SSL: float = 1e-4           # SSL pretraining LR
    WEIGHT_DECAY: float = 1e-2
    GRAD_CLIP: float = 1.0
    WARMUP_FRAC: float = 0.05

    EPOCHS_SSL: int = 100
    EPOCHS_FINETUNE: int = 80
    EARLY_STOP_PATIENCE: int = 15
    AMP: bool = True

    # EMA / SWA
    EMA_DECAY: float = 0.999
    SWA_START_FRAC: float = 0.80
    SWA_LR: float = 1e-5

    # TTA
    TTA_PASSES: int = 5

    # ------------------------------------------------------------------ evaluation
    BOOTSTRAP_ITERS: int = 1000
    BOOTSTRAP_CI: float = 0.95
    N_ENSEMBLE_SEEDS: int = 5      # number of random-seed runs to ensemble

    # ------------------------------------------------------------------ plotting
    PLOT_DPI: int = 300
    PLOT_FMT: Tuple[str, ...] = ("png", "pdf")

    # ------------------------------------------------------------------ device
    @staticmethod
    def device() -> str:
        import torch
        if torch.cuda.is_available():
            try:
                torch.tensor([1.0], device="cuda")
                return "cuda"
            except Exception:
                return "cpu"
        return "cpu"


CFG = _CFG()


def summary() -> str:
    lines = ["CardioFusion-SSL — Configuration Summary", "=" * 50]
    for k in sorted(vars(CFG).keys()):
        v = getattr(CFG, k)
        if isinstance(v, Path):
            v = str(v)
        lines.append(f"  {k:<30} = {v}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
