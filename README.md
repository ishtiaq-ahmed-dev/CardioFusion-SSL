<div align="center">

# CardioFusion-SSL

### Multimodal ECG + PCG cardiac screening with cross-modal self-supervision

Cross-Modal Self-Supervised Pretraining and Hierarchical Fusion for Multimodal ECG–PCG Cardiac Screening with Leakage-Free Cross-Dataset Validation

[![Paper](https://img.shields.io/badge/IEEE%20JBHI-under%20review-00629B?style=flat-square)](#citation)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-2EA44F?style=flat-square)](LICENSE)
[![AUROC](https://img.shields.io/badge/AUROC-0.985-informational?style=flat-square)](#results)

[Results](#results) · [Architecture](#architecture) · [Installation](#installation) · [Usage](#usage) · [Evaluation](#evaluation-protocol) · [Citation](#citation)

</div>

---

CardioFusion-SSL classifies heart recordings as normal or abnormal using the electrocardiogram (ECG) and phonocardiogram (PCG) together. The two signals are recorded at the same time, so each heartbeat appears once electrically and once acoustically. We pretrain the encoders to match the two views without labels, then fuse them at three time scales for classification.

On PhysioNet/CinC 2016 training-a, with **subject-disjoint 10-fold cross-validation** (no patient appears in both train and test), the model reaches:

<div align="center">

| AUROC | Sensitivity | Specificity | Balanced Acc. | MCC | ECE |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.9847** ± 0.020 | **0.9529** ± 0.050 | **0.9698** ± 0.036 | **0.9613** ± 0.029 | **0.9138** ± 0.059 | **0.028** |

</div>

Earlier multimodal results on this corpus (for example PACFNet, AUROC 0.9967) split data at the beat level, so beats from one patient end up in both train and test. Our protocol does not allow that, and the gap to that figure is 1.2 points.

## Highlights

- **Cross-modal pretraining.** Contrastive (InfoNCE) plus masked-reconstruction objectives on 12,412 paired ECG/PCG segments from CinC 2016 and EPHNOGRAM. The pretrained encoders retrieve the matching segment from the other modality 91.4% of the time (top-1), and pretraining adds about +0.038 AUROC downstream.
- **Hierarchical fusion.** Bidirectional cross-attention runs at sub-beat (~62 ms), cardiac-cycle (~250 ms) and recording (~1 s) scales, so the model can use QRS–S1 timing, beat-to-beat synchrony and overall rhythm.
- **Works with one signal missing.** Learned placeholder tokens stand in for an absent modality. The same checkpoint accepts paired, ECG-only or PCG-only input.
- **Leakage-free benchmark.** Folds are split by subject, and the splitting is fixed by seed so results can be reproduced.
- **External validation on ten public datasets.** Four PCG and six ECG corpora, reported in full, including the datasets where the model does poorly.

## Results

### Internal: subject-disjoint 10-fold CV (CinC 2016 training-a)

Threshold set at the Youden-optimal point. Predictions use test-time augmentation and temperature scaling (T = 0.73). AUROC and AUPRC do not depend on the threshold.

| Metric | Mean ± SD | 95% CI |
|:---|:---:|:---:|
| **AUROC** | **0.9847 ± 0.0198** | [0.971, 0.999] |
| AUPRC | 0.9486 ± 0.0899 | [0.884, 1.000] |
| Sensitivity | 0.9529 ± 0.0502 | [0.917, 0.989] |
| Specificity | 0.9698 ± 0.0357 | [0.944, 0.995] |
| Balanced accuracy | 0.9613 ± 0.0293 | [0.940, 0.982] |
| Macro F1 | 0.9559 ± 0.0308 | [0.934, 0.978] |
| Accuracy | 0.9663 ± 0.0257 | [0.948, 0.985] |
| MCC | 0.9138 ± 0.0593 | [0.871, 0.956] |
| ECE (calibrated) | 0.0275 | — |

### Ablation

Architecture ablations use the base recipe (contrastive-only pretraining, no augmentation, no TTA), which explains the lower reference AUROC.

| Variant | AUROC | Δ |
|:---|:---:|:---:|
| **Full model** | **0.9214** | — |
| Without SSL pretraining | 0.8830 | −0.0384 |
| Single-scale fusion (s = 16) | 0.9207 | −0.0007 |
| Early fusion (concatenation) | 0.9126 | −0.0088 |
| ECG only | 0.8633 | −0.0581 |
| PCG only | 0.6350 | −0.2864 |

The full training recipe (masked reconstruction, SpecAugment, MixUp, modality dropout, SWA, TTA and temperature scaling) raises AUROC from 0.9214 to 0.9847.

### External validation (10-fold soft-vote ensemble, single modality)

| Dataset | Modality | AUROC | F1 | Sensitivity |
|:---|:---|:---:|:---:|:---:|
| CinC 2016 validation | PCG | 0.603 | 0.530 | 0.323 |
| CinC 2016 training-b–f | PCG | 0.562 | 0.113 | 0.000 |
| CirCor DigiScope (paediatric) | PCG | 0.487 | 0.343 | 0.003 |
| BMD-HS (valvular) | PCG | 0.453 | 0.375 | 0.348 |
| CPSC-2018 | 12-lead ECG | **0.690** | 0.210 | 0.143 |
| PTB-XL | 12-lead ECG | 0.615 | 0.426 | 0.135 |
| Georgia (CinC 2021) | 12-lead ECG | 0.632 | 0.338 | 0.228 |
| Chapman-Shaoxing | 12-lead ECG | 0.535 | 0.298 | 0.174 |
| MIT-BIH Arrhythmia | 2-lead ECG | n/a¹ | 0.091 | 0.101 |
| Ningbo (CinC 2021) | 12-lead ECG | 0.328² | 0.179 | 0.203 |

<sub>¹ Every record is predicted positive, so AUROC is undefined (label-mapping gap). ² Below chance because 98.4% of Ningbo records are abnormal and the threshold is fixed; prior-adjusted thresholding recovers this.</sub>

These numbers are a large drop from the internal results. A model trained on paired signals from 405 subjects does not transfer well to single-modality data from other devices, populations and label definitions. We report the results in full rather than leave them out.

## Architecture

20.24 M parameters in total.

```
      ECG encoder · 4.75 M               PCG encoder · 4.83 M
      ────────────────────               ────────────────────
      patch stem (k = 25)                log-mel spectrogram (128 × 111)
      sinusoidal positions               2-D patch embedding (16 × 16)
      6 × Mamba / Transformer            6 × AST Transformer
      80 tokens × 256                    48 tokens × 256
                 │                                  │
                 └────────────────┬─────────────────┘
                                  ▼
                 Hierarchical fusion · 9.48 M
                 bidirectional cross-attention, scales {4, 16, 64}
                                  │
                                  ▼
                 1536-d joint representation
                                  │
                                  ▼
                 Head · 1.18 M   Linear → LN → GELU → Dropout → Linear
                                  │
                                  ▼
                           Normal / Abnormal
```

When a modality is missing, the tokens `m_ecg` and `m_pcg` take its place. The ECG encoder uses `mamba-ssm` if it is installed and otherwise uses a Transformer.

## Repository layout

```
configs/cfg.py              all hyperparameters and paths
data/adapter.py             cache loader, PairedCacheDataset
models/
  mamba_ecg.py              ECG encoder
  ast_pcg.py                PCG encoder
  hier_fusion.py            multi-scale bidirectional fusion
  ssl_heads.py              projection and reconstruction heads
  full_model.py             end-to-end model
pretraining/contrastive.py  cross-modal InfoNCE loss
utils/
  augment.py                SpecAugment, MixUp, modality dropout, soft focal loss
  posthoc.py                temperature scaling, ensemble weighting
  stats.py                  DeLong, McNemar, Wilcoxon, ECE
  visualise.py              figures
scripts/                    entry points (see Usage)
DATASETS.md                 download links and label mappings
```

## Installation

You need Python 3.11, an NVIDIA GPU with at least 8 GB of memory (16 GB recommended) and CUDA 12.x.

```bash
git clone https://github.com/ishtiaq-ahmed-dev/CardioFusion-SSL.git
cd CardioFusion-SSL
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1

# PyTorch first (nightly cu128 is needed for RTX 50-series)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt

# optional, Linux only: Mamba ECG encoder
pip install mamba-ssm
```

## Data

All twelve datasets are public. [DATASETS.md](DATASETS.md) lists download links, the expected folder layout and how each label set is mapped to normal/abnormal. You need two of them for training:

```bash
python -c "import wfdb; wfdb.dl_database('challenge-2016', dl_dir='<DATASETS_ROOT>/challenge-2016/1.0.0')"
python -c "import wfdb; wfdb.dl_database('ephnogram',      dl_dir='<DATASETS_ROOT>/ephnogram')"
```

Then set `DATASETS_ROOT` in [`configs/cfg.py`](configs/cfg.py).

## Usage

```bash
# 0. sanity check: forward and backward pass on a synthetic batch
python -m scripts.smoke_test

# 1. cross-modal pretraining (contrastive + masked reconstruction, 200 epochs)
python -m scripts.pretrain_v2

# 2. fine-tuning, 10 subject-disjoint folds run in parallel
python -m scripts.run_parallel_folds_v2            # --seeds 3 for a 30-model ensemble

# 3. test-time augmentation on the internal folds
python -m scripts.tta_internal_v2

# 4. external validation
python -m scripts.run_external_eval  --ckpt-dir checkpoints_v2 --ckpt-glob "fold_*_seed_*_best.pt" --out-suffix _v2
python -m scripts.eval_new_datasets  --ckpt-dir checkpoints_v2 --ckpt-glob "fold_*_seed_*_best.pt" --out-suffix _v2

# 5. ablations, statistics, figures
python -m scripts.ablation
python -m scripts.run_stats_tests
python -c "from utils.visualise import generate_all; generate_all()"
```

`python -m scripts.run_tier23_pipeline` runs steps 1–4 end to end and logs to `results/tier23_pipeline.log`. The original v1 pipeline (`scripts.pretrain`, `scripts.run_parallel_folds`) is kept for the ablation baseline.

### Approximate runtimes (RTX 5070 Ti, 16 GB)

| Stage | Time |
|:---|:---:|
| Pretraining, 200 epochs | ~2 h |
| Fine-tuning, 10 folds × 1 seed | ~1.5 h |
| Fine-tuning, 10 folds × 3 seeds | ~4 h |
| Internal TTA | ~50 min |
| External validation | ~1 h |
| Ablations (5 variants) | ~7 h |

All runs use `SEED = 1337`.

## Evaluation protocol

The 405 subjects in CinC 2016 training-a are stratified by label and split into ten folds **at the subject level**. Each subject contributes one recording, so no recording or beat from a test subject is seen during training.

Beat-level splitting, which several earlier papers use, puts beats from the same patient on both sides of the split. On this corpus we estimate it inflates AUROC by about 7.5 points. We report subject-disjoint results only.

External evaluation averages the predicted probabilities of all fold checkpoints. Calibration is reported as expected calibration error (ECE) over ten equal-width bins, before and after temperature scaling.

## Citation

If this code helps your work, please cite the paper and the datasets you use (see [DATASETS.md](DATASETS.md)).

```bibtex
@article{ahmed2026cardiofusion,
  title   = {CardioFusion-SSL: Cross-Modal Self-Supervised Pretraining
             and Hierarchical Fusion for Multimodal ECG-PCG Cardiac Screening
             with Leakage-Free Cross-Dataset Validation},
  author  = {Ahmed, Ishtiaq and Ambreen, Abida and Jawad, Syeda Laiba},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Under review}
}
```

## Authors

**Ishtiaq Ahmed** · lead author and maintainer
Institute of Management Sciences (IMSciences), Peshawar
[ORCID 0009-0007-5485-211X](https://orcid.org/0009-0007-5485-211X) · [@ishtiaq-ahmed-dev](https://github.com/ishtiaq-ahmed-dev) · iashtiaqkhattak@gmail.com

**Dr. Abida Ambreen, MBBS** · medical review
Khyber Medical College (KMC), Peshawar

**Syeda Laiba Jawad** · methodology and investigation
Institute of Management Sciences (IMSciences), Peshawar

Supervised by **Mr. Ali Haider**, IMSciences Peshawar.

## License

The code is released under the [MIT License](LICENSE). Each dataset keeps its original license; see [DATASETS.md](DATASETS.md).

<div align="center">
<sub>This is a research prototype and is not a medical device. Do not use it for clinical decisions.</sub>
</div>
