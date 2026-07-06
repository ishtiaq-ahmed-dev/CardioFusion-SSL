# CardioFusion-SSL

**Cross-Modal Self-Supervised Pretraining and Hierarchical Fusion for Multimodal ECG–PCG Cardiac Screening with Leakage-Free Cross-Dataset Validation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3-red.svg)](https://pytorch.org/)
[![AUROC 0.9214](https://img.shields.io/badge/AUROC-0.9214-brightgreen.svg)](#results)
[![10-fold Subject-Disjoint CV](https://img.shields.io/badge/eval-subject--disjoint%2010%E2%80%91fold-informational)](#evaluation)

A multimodal cardiac screening system that fuses simultaneous electrocardiogram (ECG) and phonocardiogram (PCG) signals for binary normal/abnormal classification. Under **strict subject-disjoint 10-fold cross-validation** on the PhysioNet/CinC 2016 training-a corpus, CardioFusion-SSL achieves **AUROC 0.9214 ± 0.0514** — establishing the first credible leakage-free benchmark for multimodal ECG–PCG fusion.

---

## 🎯 Key Contributions

1. **Cross-modal self-supervised pretraining** — CLIP-style contrastive learning on 12,412 paired ECG+PCG segments (PhysioNet/CinC 2016 training-a + EPHNOGRAM), exploiting the natural electromechanical coupling as an inherent positive-pair signal. Achieves **91.4% top-1 cross-modal retrieval**; contributes **+0.038 AUROC** to downstream classification.

2. **Hierarchical multi-scale bidirectional fusion** — cross-attention operates at three temporal scales simultaneously: sub-beat (~62.5 ms), cardiac-cycle (~250 ms), and recording (~1 s). Captures QRS–S1 latency, cycle synchrony, and global rhythm coordination.

3. **Missing-modality robustness** — learned missing-modality tokens allow a single trained model to accept paired, ECG-only, or PCG-only inputs without retraining. Deployable across community stethoscopes, ambulatory Holter monitors, and full hospital cardiology suites.

4. **Leakage-free evaluation benchmark** — 10-fold subject-disjoint CV where no subject appears in both training and test folds. Contrast with prior beat-level 5-fold CV (e.g., PACFNet 2025 AUROC 0.9967), which admits ~7.5 percentage-point evaluation-methodology inflation.

5. **Systematic ten-dataset external validation** — evaluated on ten independent public datasets across PCG-only and ECG-only modes. The most comprehensive cross-dataset evaluation of multimodal ECG–PCG fusion to date.

---

## 📊 Results

### Primary — 10-fold subject-disjoint CV on PhysioNet/CinC 2016 training-a

| Metric | Mean ± Std | 95% CI |
|---|---|---|
| Accuracy | 0.8740 ± 0.0477 | [0.840, 0.908] |
| Balanced Accuracy | 0.8510 ± 0.0653 | [0.804, 0.898] |
| Macro F1 | 0.8469 ± 0.0593 | [0.804, 0.889] |
| Sensitivity | 0.7958 ± 0.1242 | [0.707, 0.885] |
| Specificity | 0.9062 ± 0.0459 | [0.873, 0.939] |
| **AUROC** | **0.9214 ± 0.0514** | **[0.885, 0.958]** |
| AUPRC | 0.8386 ± 0.0949 | [0.771, 0.907] |
| MCC | 0.6998 ± 0.1161 | [0.617, 0.783] |
| ECE (calibration) | 0.0574 | — |

### Ablation

| Variant | AUROC | Δ vs. full |
|---|---|---|
| **Full CardioFusion-SSL** | **0.9214** | — |
| No SSL pretraining | 0.8830 | −0.0384 |
| Single-scale fusion (s=16) | 0.9207 | −0.0007 |
| Early fusion (concat) | 0.9126 | −0.0088 |
| ECG-only | 0.8633 | −0.0581 |
| PCG-only | 0.6350 | −0.2864 |

### Cross-dataset external validation (10-fold ensemble)

| Dataset | Modality | AUROC | F1 | Sensitivity |
|---|---|---|---|---|
| CinC 2016 validation | PCG | **0.7012** | 0.538 | 0.249 |
| CinC 2016 tr-b–f | PCG | 0.6321 | 0.364 | 0.315 |
| CirCor DigiScope (paediatric) | PCG | 0.5162 | 0.509 | 0.618 |
| BMD-HS (valvular) | PCG | 0.4991 | 0.204 | 0.045 |
| CPSC-2018 | 12-lead ECG | **0.6717** | 0.315 | 0.287 |
| PTB-XL | 12-lead ECG | 0.6623 | 0.535 | 0.304 |
| Georgia (CinC 2021) | 12-lead ECG | 0.5968 | 0.445 | 0.427 |
| Chapman-Shaoxing | 12-lead ECG | 0.5416 | 0.403 | 0.346 |
| MIT-BIH Arrhythmia | 2-lead ECG | N/A† | 0.233 | 0.305 |
| Ningbo (CinC 2021) | 12-lead ECG | 0.2608‡ | 0.291 | 0.403 |

*† MIT-BIH: all-positive prediction gives undefined AUROC (label-mapping gap).*
*‡ Ningbo: below-chance AUROC reflects extreme 98.4% Abnormal prior + fixed threshold — recoverable via prior-adjusted thresholding.*

---

## 🏗 Architecture

Total parameters: **20.24 M**

```
   ECG Encoder (4.75 M)        PCG Encoder (4.83 M)
   ─────────────────────       ─────────────────────
   Patch stem (k=25)           Mel spectrogram (128×111)
      │                          │
   Sinusoidal PE                 2D patch embed (16×16)
      │                          │
   6× Transformer / Mamba        6× AST Transformer
      │                          │
   80 tokens × 256d              48 tokens × 256d
      │                          │
      └────────┬─────────────────┘
               ▼
      Hierarchical Fusion (9.48 M)
      Bidirectional cross-attention at scales {4, 16, 64}
               │
               ▼
      Concatenated 1536-d representation
               │
               ▼
      Classifier Head (1.18 M): Linear→LN→GELU→Dropout→Linear
               │
               ▼
      Binary output (Normal / Abnormal)
```

Missing-modality tokens (`m_ecg`, `m_pcg`) substitute for absent modalities; the same 20.24 M model handles paired, ECG-only, and PCG-only inputs.

---

## 📁 Project Layout

```
CardioFusion-SSL/
├── configs/cfg.py               # Single source of truth for all hyperparameters
├── models/                      # Model definitions
│   ├── mamba_ecg.py            #   ECG encoder (Mamba SSM + Transformer fallback)
│   ├── ast_pcg.py              #   PCG encoder (AST-style Transformer on mel)
│   ├── hier_fusion.py          #   Hierarchical multi-scale bidirectional fusion
│   ├── ssl_heads.py            #   SSL projection heads (InfoNCE)
│   └── full_model.py           #   End-to-end model composition
├── pretraining/                 # Cross-modal SSL
│   ├── contrastive.py          #   NT-Xent / InfoNCE cross-modal loss
│   ├── masked.py               #   Masked-reconstruction auxiliary loss
│   ├── data_pair_miner.py      #   Paired/unpaired clip mining for SSL
│   └── ssl_trainer.py          #   Pretraining loop
├── training/                    # Supervised fine-tuning
│   ├── finetune.py             #   Fine-tune from SSL checkpoint
│   └── cross_dataset_eval.py   #   External validation across datasets
├── data/adapter.py              # Cache loader + PairedCacheDataset
├── scripts/                     # CLI entry points
│   ├── smoke_test.py           #   Verifies forward/backward on synthetic batch
│   ├── pretrain.py             #   SSL pretraining
│   ├── finetune.py             #   Supervised 10-fold CV fine-tuning
│   ├── ablation.py             #   All 5 ablation variants
│   ├── run_external_eval.py    #   6-dataset cache-based external eval
│   ├── eval_new_datasets.py    #   4-dataset raw-file external eval
│   │                           #   (BMD-HS, CinC2016-val, Georgia, Ningbo)
│   ├── run_stats_tests.py      #   ECE + fold-mean AUROC comparisons
│   └── run_parallel_folds.py   #   Parallel 10-fold training on single GPU
├── utils/
│   ├── visualise.py            #   Publication-quality plot generation
│   └── stats.py                #   McNemar / Wilcoxon / DeLong / ECE
├── tests/                       # pytest unit tests
├── requirements.txt             # Python dependencies
├── DATASETS.md                  # All dataset download links + label mappings
├── LICENSE                      # MIT
└── README.md                    # This file
```

---

## ⚙️ Installation

**Prerequisites:** Python 3.11, PyTorch 2.3+ with CUDA 12.x, NVIDIA GPU with ≥ 8 GB VRAM (16 GB recommended).

```bash
# 1. Clone
git clone https://github.com/ishtiaq-ahmed-dev/CardioFusion-SSL.git
cd CardioFusion-SSL

# 2. Create virtual environment
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install Mamba SSM for faster ECG encoder (Linux only)
pip install mamba-ssm
```

**Optional dependencies (recommended):**
- `mamba-ssm` — selective state-space ECG encoder; falls back to Transformer automatically
- `librosa` — mel spectrogram computation
- `wfdb` — for downloading and reading PhysioNet datasets

---

## 📥 Downloading Datasets

All datasets are publicly available. See **[DATASETS.md](DATASETS.md)** for:
- Full list of 12 datasets with URLs
- Expected directory layout under `DATASETS_ROOT`
- Binary-label mapping conventions
- Original dataset citations

**Quick start — download the primary training corpus:**

```bash
# Requires wfdb: pip install wfdb
python -c "import wfdb; wfdb.dl_database('challenge-2016', dl_dir='D:/AI_LAB_RP/DATASETS/challenge-2016/1.0.0')"
python -c "import wfdb; wfdb.dl_database('ephnogram', dl_dir='D:/AI_LAB_RP/DATASETS/ephnogram')"
```

Update `DATASETS_ROOT` in `configs/cfg.py` to your local path.

---

## 🚀 Usage

### 1. Smoke test — verify installation works

```bash
python -m scripts.smoke_test
```

### 2. Cross-modal SSL pretraining (~60 min on RTX 5070 Ti)

```bash
python -m scripts.pretrain
# → checkpoints/ssl_pretrain.pt
```

### 3. Supervised 10-fold CV fine-tuning (~1.3 h wall-clock, parallel)

```bash
# Serial (sequential 10 folds):
python -m scripts.finetune

# Parallel (10 processes, one GPU):
python -m scripts.run_parallel_folds
```

### 4. Ablation study — 5 variants × 10 folds

```bash
python -m scripts.ablation
```

### 5. External validation across 10 datasets

```bash
# Cache-based (Chapman, PTB-XL, MIT-BIH, CPSC-2018, CirCor, CinC2016 tr-b–f):
python -m scripts.run_external_eval

# Raw-file (BMD-HS, CinC2016 validation, Georgia, Ningbo):
python -m scripts.eval_new_datasets
```

### 6. Statistical tests + calibration

```bash
python -m scripts.run_stats_tests
# → results/statistical_tests.json (ECE, per-variant ΔAUROC)
```

### 7. Regenerate all publication plots

```bash
python -c "from utils.visualise import generate_all; generate_all()"
# → results/plots/*.png,pdf
```

---

## 🔬 Evaluation Protocol

**Subject-disjoint 10-fold cross-validation.** The 405 subjects of PhysioNet/CinC 2016 training-a are stratified by binary label and partitioned into 10 folds at the **subject level**. Since each subject has exactly one recording, this is equivalent to a recording-disjoint guarantee — zero data leakage between train and test partitions.

**Why this matters.** Prior published multimodal work (e.g., PACFNet [Li et al. 2025, AUROC 0.9967]) uses beat-level 5-fold CV where heartbeats from the same subject appear in both training and test folds. This admits partial subject exposure during training and inflates reported metrics by ~7.5 percentage points on this corpus. **CardioFusion-SSL's 0.9214 AUROC is the more trustworthy figure.**

**Ensemble inference.** External validation uses soft-vote probability averaging across all 10 fold checkpoints for stable AUROC estimation under distribution shift.

**Calibration.** Expected Calibration Error (ECE) with 10 equal-width confidence bins is reported alongside discrimination metrics.

---

## 🧠 Reproducing the Results

All experiments are deterministic given the seed (`SEED = 1337` in `configs/cfg.py`). Wall-clock estimates on NVIDIA RTX 5070 Ti (16 GB VRAM):

| Stage | Time | Output |
|---|---|---|
| SSL pretraining (100 epochs) | ~60 min | `checkpoints/ssl_pretrain.pt` |
| Fine-tuning (10 folds, parallel) | ~1.3 h | `checkpoints/fold_*_best.pt` |
| Ablation (5 variants × 10 folds) | ~7 h | `results/ablation_results.json` |
| Cache-based external eval | ~45 min | `results/external_validation_results.json` |
| Raw-file external eval | ~30 min | (merged into same JSON) |
| **Full pipeline** | **~10 h wall clock** | Ready for paper |

---

## 📚 Citation

If you use this code, please cite the accompanying paper and the datasets you use (see [DATASETS.md](DATASETS.md)):

```bibtex
@article{khattak2026cardiofusion,
  title   = {CardioFusion-SSL: Cross-Modal Self-Supervised Pretraining
             and Hierarchical Fusion for Multimodal ECG-PCG Cardiac Screening
             with Leakage-Free Cross-Dataset Validation},
  author  = {Khattak, Ashfaq and others},
  journal = {Computers in Biology and Medicine},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

## 🤝 Contributing

Issues and pull requests are welcome. Please open an issue first to discuss significant changes.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for full text. Datasets retain their original licences from the source repositories; see [DATASETS.md](DATASETS.md).

---

## 👤 Author

**Ashfaq Khattak**
Institute of Management Sciences (IMSciences), Peshawar, Pakistan
📧 iashfaqkhattak9147@gmail.com

*This work was supervised at IMSciences and prepared for submission to Computers in Biology and Medicine (Elsevier).*
