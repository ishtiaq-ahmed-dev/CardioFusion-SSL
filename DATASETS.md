# Datasets Used in CardioFusion-SSL

All datasets used in this project are **publicly available**. We do not redistribute
any raw signals — please download from the official sources below and cite the
original dataset papers.

---

## Dataset Directory Layout Expected by the Code

By default, `configs/cfg.py` points to:

```
DATASETS_ROOT = D:/AI_LAB_RP/DATASETS
```

Change this to your local path. The expected sub-directory structure:

```
DATASETS/
├── challenge-2016/1.0.0/                # PhysioNet CinC 2016
│   ├── training-a/                       # ← paired ECG+PCG (primary training)
│   ├── training-b/ ... training-f/       # ← external PCG-only validation
│   └── validation/                       # ← held-out validation set
├── ephnogram/                            # EPHNOGRAM paired ECG+PCG (SSL only)
├── circor-heart-sound/1.0.3/             # CirCor DigiScope PCG
├── bmd-hs-dataset/                       # BMD-HS valvular disease PCG
│   └── BMD-HS-Dataset-main/
│       ├── train.csv
│       └── train/                        # WAV files
├── challenge-2021/1.0.3/training/        # PhysioNet CinC 2021
│   ├── georgia/                          # Emory Healthcare 12-lead ECG
│   ├── ningbo/                           # Ningbo First Hospital 12-lead ECG
│   ├── chapman_shaoxing/
│   └── ptb_xl/
├── chapman-shaoxing/                     # Chapman-Shaoxing 12-lead ECG
├── ptb-xl/                               # PTB-XL 12-lead ECG
├── mit-bih-arrhythmia-database-1.0.0/    # MIT-BIH 2-lead ECG
└── cpsc2018/                             # CPSC-2018 12-lead ECG Challenge
```

---

## Primary Training Corpus

### 1. PhysioNet/CinC Challenge 2016 — training-a
- **Role:** Supervised fine-tuning + SSL pretraining
- **Modalities:** Simultaneous single-lead ECG + PCG @ 2000 Hz
- **Size:** 405 recordings, 405 unique subjects (11,722 4-s windows)
- **Labels:** Binary Normal / Abnormal
- **URL:** https://physionet.org/content/challenge-2016/1.0.0/
- **Paper:** Liu C. et al., *Physiol. Meas.* 37(12):2181–2213, 2016. DOI:[10.1088/0967-3334/37/12/2181](https://doi.org/10.1088/0967-3334/37/12/2181)

### 2. EPHNOGRAM
- **Role:** SSL pretraining only (unlabelled paired ECG+PCG)
- **Modalities:** Simultaneous ECG + PCG from 24 healthy adult subjects
- **Size:** 690 4-s paired windows after preprocessing
- **URL:** https://physionet.org/content/ephnogram/1.0.0/
- **Paper:** Kazemnejad A. et al., *Physiol. Meas.* 45(5):055005, 2024. DOI:[10.1088/1361-6579/ad43af](https://doi.org/10.1088/1361-6579/ad43af)

---

## External Validation — PCG-only Mode

### 3. PhysioNet/CinC 2016 training-b through training-f
- **Role:** Cross-site PCG-only external validation (52,207 windows)
- **URL:** https://physionet.org/content/challenge-2016/1.0.0/ (same as #1, different subsets)

### 4. PhysioNet/CinC 2016 official validation set
- **Role:** Balanced held-out validation (301 recordings, 2,757 windows)
- **Labels:** REFERENCE.csv: `1` = Normal, `-1` = Abnormal
- **URL:** https://physionet.org/content/challenge-2016/1.0.0/validation/

### 5. CirCor DigiScope Phonocardiogram Dataset (PhysioNet 2022)
- **Role:** Paediatric external validation (63,478 windows, 3,163 recordings, 942 subjects)
- **Population:** Paediatric Brazilian cohort
- **URL:** https://physionet.org/content/circor-heart-sound/1.0.3/
- **Papers:**
  - Oliveira J. et al., *IEEE J. Biomed. Health Inform.* 26(6):2524–2535, 2022. DOI:[10.1109/JBHI.2021.3137048](https://doi.org/10.1109/JBHI.2021.3137048)
  - Reyna M.A. et al., *Physiol. Meas.* 44(3):035006, 2023. DOI:[10.1088/1361-6579/ac7c44](https://doi.org/10.1088/1361-6579/ac7c44)

### 6. BMD-HS: BUET Multi-disease Heart Sound Dataset
- **Role:** Bangladeshi valvular disease external validation (7,772 windows, 864 recordings, 108 subjects)
- **Pathologies:** Aortic Stenosis / Regurgitation, Mitral Stenosis / Regurgitation
- **URL:** https://github.com/toufiqmusah/BMD-HS-Dataset
- **License:** As specified in the source repository — verify before commercial use

---

## External Validation — ECG-only Mode

### 7. Chapman-Shaoxing 12-lead ECG
- **Role:** ECG-only external validation (315,735 windows, 45,152 recordings)
- **URL:** https://physionet.org/content/ecg-arrhythmia/1.0.0/
- **Paper:** Zheng J. et al., *Sci. Data* 7:48, 2020. DOI:[10.1038/s41597-020-0386-x](https://doi.org/10.1038/s41597-020-0386-x)

### 8. PTB-XL
- **Role:** ECG-only external validation (152,593 windows, 21,837 recordings)
- **URL:** https://physionet.org/content/ptb-xl/1.0.3/
- **Paper:** Wagner P. et al., *Sci. Data* 7:154, 2020. DOI:[10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6)

### 9. MIT-BIH Arrhythmia Database
- **Role:** ECG-only external validation (86,544 windows, 48 recordings)
- **URL:** https://physionet.org/content/mitdb/1.0.0/
- **Paper:** Moody G.B. & Mark R.G., *IEEE Eng. Med. Biol. Mag.* 20(3):45–50, 2001. DOI:[10.1109/51.932724](https://doi.org/10.1109/51.932724)

### 10. CPSC-2018 12-lead ECG Challenge
- **Role:** ECG-only external validation (134,802 windows, 6,877 recordings)
- **URL:** http://2018.icbeb.org/Challenge.html
- **Paper:** Liu F. et al., *J. Med. Imaging Health Inform.* 8(7):1368–1373, 2018. DOI:[10.1166/jmihi.2018.2442](https://doi.org/10.1166/jmihi.2018.2442)

### 11. PhysioNet/CinC 2021 — Georgia subset
- **Role:** ECG-only external validation (41,172 windows, 10,332 recordings)
- **Source:** Emory Healthcare, Atlanta, USA
- **URL:** https://physionet.org/content/challenge-2021/1.0.3/

### 12. PhysioNet/CinC 2021 — Ningbo subset
- **Role:** ECG-only external validation (26,244 windows, 6,561 recordings)
- **Source:** Ningbo First Hospital, China
- **URL:** https://physionet.org/content/challenge-2021/1.0.3/

**CinC 2021 paper:** Reyna M.A. et al., *Comput. Cardiol.* 48:1–4, 2021.
DOI:[10.23919/CinC53138.2021.9662687](https://doi.org/10.23919/CinC53138.2021.9662687)

---

## Label Mapping to Binary Normal / Abnormal

All datasets are mapped to a unified binary label scheme for our screening task:

| Dataset | Native label | → Binary mapping |
|---|---|---|
| CinC 2016 (all subsets) | Challenge label −1 / +1 | `+1` → 0 (Normal), `−1` → 1 (Abnormal) |
| CinC 2016 validation | REFERENCE.csv `1` / `-1` | Same as above |
| CirCor DigiScope | Absent / Present / Unknown | Absent → 0; Present/Unknown → 1 |
| BMD-HS | N=1 / N=0 / N=N/A | 1 → 0; 0 → 1; N/A → skip |
| CinC 2021 (Georgia/Ningbo) | SNOMED-CT codes | `426783006` or `164865005` → 0; else 1 |
| PTB-XL | SCP codes | `NORM` → 0; else 1 |
| Chapman-Shaoxing | Rhythm categories | Sinus rhythm → 0; else 1 |
| MIT-BIH | Beat annotations | Normal beat (`N`) → 0; else 1 |
| CPSC-2018 | 9 SNOMED codes | `Normal` → 0; else 1 |

---

## Download Automation

All PhysioNet datasets can be downloaded with the WFDB Python package:

```bash
pip install wfdb
python -c "import wfdb; wfdb.dl_database('challenge-2016', dl_dir='D:/AI_LAB_RP/DATASETS/challenge-2016/1.0.0')"
```

Or via the PhysioNet mirror:

```bash
wget -r -N -c -np https://physionet.org/files/challenge-2016/1.0.0/
```

Non-PhysioNet datasets (BMD-HS, CPSC-2018) must be downloaded from their
respective source repositories (URLs above).

---

## Citation Requirements

If you use this project, please cite:

1. **Our paper** (see `CITATION.cff` in this repository)
2. **The original dataset papers** listed above for every dataset you use.
