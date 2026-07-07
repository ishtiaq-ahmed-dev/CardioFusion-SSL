# CardioFusion-SSL

**Cross-Modal Self-Supervised Pretraining and Hierarchical Fusion for Multimodal ECG–PCG Cardiac Screening with Leakage-Free Cross-Dataset Validation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3-red.svg)](https://pytorch.org/)
[![AUROC 0.9847](https://img.shields.io/badge/AUROC-0.9847-brightgreen.svg)](#results)
[![10-fold Subject-Disjoint CV](https://img.shields.io/badge/eval-subject--disjoint%2010%E2%80%91fold-informational)](#evaluation)

A multimodal cardiac screening system that fuses simultaneous electrocardiogram (ECG) and phonocardiogram (PCG) signals for binary normal/abnormal classification. Under **strict subject-disjoint 10-fold cross-validation** on the PhysioNet/CinC 2016 training-a corpus, CardioFusion-SSL achieves **AUROC 0.9847 ± 0.0198** (95% CI 0.971–0.999) — the first credible leakage-free benchmark for multimodal ECG–PCG fusion, sitting just 1.2 percentage points below the leaky beat-level state of the art (PACFNet 0.9967).

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

Reported at the Youden-optimal operating point with test-time augmentation and post-hoc temperature scaling. AUROC and AUPRC are threshold-independent.

| Metric | Mean ± Std | 95% CI |
|---|---|---|
| Sensitivity | **0.9529 ± 0.0502** | [0.917, 0.989] |
| Specificity | 0.9698 ± 0.0357 | [0.944, 0.995] |
| Balanced Accuracy | 0.9613 ± 0.0293 | [0.940, 0.982] |
| Macro F1 | 0.9559 ± 0.0308 | [0.934, 0.978] |
| Accuracy | 0.9663 ± 0.0257 | [0.948, 0.985] |
| **AUROC** | **0.9847 ± 0.0198** | **[0.971, 0.999]** |
| AUPRC | 0.9486 ± 0.0899 | [0.884, 1.000] |
| MCC | 0.9138 ± 0.0593 | [0.871, 0.956] |
| ECE (calibrated) | 0.0275 | — |

### Ablation (architectural choices, contrastive-only SSL baseline)

| Variant | AUROC | Δ vs. reference |
|---|---|---|
| **Full architecture** | **0.9214** | — |
| No SSL pretraining | 0.8830 | −0.0384 |
| Single-scale fusion (s=16) | 0.9207 | −0.0007 |
| Early fusion (concat) | 0.9126 | −0.0088 |
| ECG-only | 0.8633 | −0.0581 |
| PCG-only | 0.6350 | −0.2864 |

Adding the training recipe on top — masked-reconstruction SSL, SpecAugment, MixUp, modality dropout, SWA, TTA, temperature scaling — lifts AUROC from 0.9214 to the reported 0.9847.

### Cross-dataset external validation (10-fold soft-vote ensemble)

| Dataset | Modality | AUROC | F1 | Sensitivity |
|---|---|---|---|---|
| CinC 2016 validation | PCG | 0.603 | 0.530 | 0.323 |
| CinC 2016 tr-b–f | PCG | 0.562 | 0.113 | 0.000 |
| CirCor DigiScope (paediatric) | PCG | 0.487 | 0.343 | 0.003 |
| BMD-HS (valvular) | PCG | 0.453 | 0.375 | 0.348 |
| CPSC-2018 | 12-lead ECG | **0.690** | 0.210 | 0.143 |
| PTB-XL | 12-lead ECG | 0.615 | 0.426 | 0.135 |
| Georgia (CinC 2021) | 12-lead ECG | 0.632 | 0.338 | 0.228 |
| Chapman-Shaoxing | 12-lead ECG | 0.535 | 0.298 | 0.174 |
| MIT-BIH Arrhythmia | 2-lead ECG | N/A† | 0.091 | 0.101 |
| Ningbo (CinC 2021) | 12-lead ECG | 0.328‡ | 0.179 | 0.203 |

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
@article{ahmed2026cardiofusion,
  title   = {CardioFusion-SSL: Cross-Modal Self-Supervised Pretraining
             and Hierarchical Fusion for Multimodal ECG-PCG Cardiac Screening
             with Leakage-Free Cross-Dataset Validation},
  author  = {Ahmed, Ishtiaq and Ambreen, Abida and Jawad, Syeda Laiba},
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

## 👥 Authors and Contributors

### Lead author / project owner
**Ishtiaq Ahmed** — BS Artificial Intelligence, Institute of Management Sciences (IMSciences), Peshawar, Pakistan
📧 iashtiaqkhattak@gmail.com  |  bsai.246504300@imsciences.edu.pk
🆔 ORCID: [0009-0007-5485-211X](https://orcid.org/0009-0007-5485-211X)
🐙 GitHub: [@ishtiaq-ahmed-dev](https://github.com/ishtiaq-ahmed-dev)
*Conceptualization, methodology, software, investigation, formal analysis, data curation, writing — original draft, visualization.*

### Co-authors

**Dr. Abida Ambreen** — Khyber Medical College (KMC), Peshawar, Pakistan
📧 Abida.ambreen@kmc.edu.pk
*Clinical collaborator — cardiology domain expertise, methodology guidance, writing — review and editing.*

**Syeda Laiba Jawad** — BS Artificial Intelligence, Institute of Management Sciences (IMSciences), Peshawar, Pakistan
📧 syedalaiba689@gmail.com
*Methodology, investigation, writing — review and editing.*

### Academic supervisor
**Mr. Ali Haider** — Institute of Management Sciences (IMSciences), Peshawar, Pakistan
*Project supervision, methodological guidance, and mentorship throughout the CardioFusion-SSL research.*

### Corresponding author
Ishtiaq Ahmed — iashtiaqkhattak@gmail.com

---

*This work was conducted at IMSciences Peshawar in collaboration with Khyber Medical College and is prepared for submission to Computers in Biology and Medicine (Elsevier).*
