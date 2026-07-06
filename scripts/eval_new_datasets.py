"""Evaluate CardioFusion-SSL ensemble on new datasets (no cache builder needed).

Processes raw signal files directly, applies identical preprocessing to main
pipeline, runs 10-fold soft-vote ensemble, and appends results to
external_validation_results.json.

New datasets handled:
  PCG-only (ECG=missing):
    - BMD-HS  (D:/AI_LAB_RP/DATASETS/bmd-hs-dataset)
    - CinC2016 validation set (D:/AI_LAB_RP/DATASETS/challenge-2016/1.0.0/validation)
  ECG-only (PCG=missing):
    - Georgia 12-lead ECG  (challenge-2021/1.0.3/training/georgia)
    - Ningbo 12-lead ECG   (challenge-2021/1.0.3/training/ningbo)

Usage:
    cd D:\AI_LAB_RP\CardioFusion-SSL
    python -m scripts.eval_new_datasets
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.io
import scipy.io.wavfile as wavfile
import scipy.signal as ssignal
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

from configs import CFG
from models.full_model import CardioFusionSSL
from scripts.finetune import compute_metrics

warnings.filterwarnings("ignore", category=scipy.io.matlab.MatReadWarning
                        if hasattr(scipy.io.matlab, "MatReadWarning") else UserWarning)

# ─── SNOMED-CT codes that map to Normal ───────────────────────────────────────
_NORMAL_SNOMED = {
    "426783006",   # Sinus rhythm / Normal sinus rhythm
    "164865005",   # Normal ECG
    "426177001",   # Sinus bradycardia (borderline — keep as normal in CinC 2021)
    "427084000",   # Sinus tachycardia (borderline — keep as normal)
    "733534002",   # Sinus rhythm (alternate code)
}
# For stricter mapping: only 426783006 is truly "Normal" — anything else = abnormal
_STRICT_NORMAL = {"426783006", "164865005"}


# ════════════════════════════════════════════════════════════════════════════════
#  SIGNAL PREPROCESSING  (identical parameters to main pipeline)
# ════════════════════════════════════════════════════════════════════════════════

def _butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = ssignal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


def preprocess_ecg_windows(signal: np.ndarray, fs_orig: int) -> List[np.ndarray]:
    """Resample → bandpass → slide 4s windows → z-score.  Returns list of (ECG_LEN,) arrays."""
    sig = signal.astype(np.float32)
    # resample to 500 Hz
    if fs_orig != CFG.ECG_FS:
        n_out = int(len(sig) * CFG.ECG_FS / fs_orig)
        sig = ssignal.resample(sig, n_out).astype(np.float32)
    # bandpass 0.5–40 Hz
    b, a = _butter_bandpass(CFG.ECG_BAND[0], CFG.ECG_BAND[1], CFG.ECG_FS, CFG.ECG_FILT_ORDER)
    sig = ssignal.filtfilt(b, a, sig).astype(np.float32)
    # slide windows
    step = CFG.ECG_LEN // 2   # 50% overlap
    windows = []
    if len(sig) < CFG.ECG_LEN:
        # pad short signals
        pad = CFG.ECG_LEN - len(sig)
        sig = np.pad(sig, (0, pad))
    start = 0
    while start + CFG.ECG_LEN <= len(sig):
        w = sig[start: start + CFG.ECG_LEN].copy()
        std = w.std() + 1e-8
        w = (w - w.mean()) / std
        windows.append(w)
        start += step
    return windows


def preprocess_pcg_windows(signal: np.ndarray, fs_orig: int) -> List[np.ndarray]:
    """Resample → bandpass → amplitude-norm → slide 4s windows → log-mel.
    Returns list of (MEL_N, MEL_T) arrays."""
    try:
        import librosa
    except ImportError:
        raise RuntimeError("librosa required: pip install librosa")

    sig = signal.astype(np.float32)
    # mono if stereo
    if sig.ndim > 1:
        sig = sig[:, 0]
    # resample to PCG_FS (2000 Hz)
    if fs_orig != CFG.PCG_FS:
        n_out = int(len(sig) * CFG.PCG_FS / fs_orig)
        sig = ssignal.resample(sig, n_out).astype(np.float32)
    # bandpass 25–400 Hz
    b, a = _butter_bandpass(CFG.PCG_BAND[0], CFG.PCG_BAND[1], CFG.PCG_FS, CFG.PCG_FILT_ORDER)
    sig = ssignal.filtfilt(b, a, sig).astype(np.float32)
    # amplitude normalize (98th percentile)
    p98 = np.percentile(np.abs(sig), 98)
    if p98 > 1e-8:
        sig = sig / p98
    sig = np.clip(sig, -1.0, 1.0)
    # slide windows
    step = CFG.PCG_LEN // 2   # 50% overlap
    mel_t = 1 + (CFG.PCG_LEN - CFG.MEL_WIN) // CFG.MEL_HOP + 1
    windows = []
    if len(sig) < CFG.PCG_LEN:
        sig = np.pad(sig, (0, CFG.PCG_LEN - len(sig)))
    start = 0
    while start + CFG.PCG_LEN <= len(sig):
        w = sig[start: start + CFG.PCG_LEN].copy()
        mel = librosa.feature.melspectrogram(
            y=w, sr=CFG.PCG_FS,
            n_fft=CFG.MEL_N_FFT, hop_length=CFG.MEL_HOP, win_length=CFG.MEL_WIN,
            n_mels=CFG.MEL_N, fmin=CFG.MEL_FMIN, fmax=CFG.MEL_FMAX, power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        # ensure exact T
        t = mel_db.shape[1]
        if t < mel_t:
            mel_db = np.pad(mel_db, ((0, 0), (0, mel_t - t)))
        elif t > mel_t:
            mel_db = mel_db[:, :mel_t]
        windows.append(mel_db.astype(np.float32))
        start += step
    return windows


# ════════════════════════════════════════════════════════════════════════════════
#  DATASET LOADERS
# ════════════════════════════════════════════════════════════════════════════════

Record = Tuple[List[np.ndarray], int, str]   # (windows, label, rec_id)


def load_bmd_hs(base_dir: Path) -> List[Record]:
    """BMD-HS: PCG-only.  Binary = 0 if N==1 (Normal), 1 otherwise.  Skip N/A."""
    csv_path = base_dir / "BMD-HS-Dataset-main" / "train.csv"
    wav_dir  = base_dir / "BMD-HS-Dataset-main" / "train"
    records: List[Record] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_val = row["N"].strip()
            if n_val == "N/A":       # patient_022 (MVP, ambiguous) — skip
                continue
            label = 0 if n_val == "1" else 1   # N=1 → Normal(0), N=0 → Abnormal(1)

            for i in range(1, 9):
                rec_name = row.get(f"recording_{i}", "").strip()
                if not rec_name:
                    continue
                wav_path = wav_dir / (rec_name + ".wav")
                if not wav_path.exists():
                    continue
                try:
                    sr, data = wavfile.read(str(wav_path))
                    sig = data.astype(np.float32)
                    if sig.max() > 1.0:           # int16 → float normalise
                        sig = sig / 32768.0
                    windows = preprocess_pcg_windows(sig, sr)
                    if windows:
                        records.append((windows, label, rec_name))
                except Exception as e:
                    print(f"    WARN BMD-HS {rec_name}: {e}")
    return records


def load_cinc2016_validation(val_dir: Path) -> List[Record]:
    """CinC2016 validation: PCG-only.  Labels from REFERENCE.csv (1=Normal, -1=Abnormal)."""
    ref_path = val_dir / "REFERENCE.csv"
    labels: Dict[str, int] = {}
    with open(ref_path, newline="", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                raw_lbl = int(parts[1])
                labels[parts[0]] = 0 if raw_lbl == 1 else 1   # 1→Normal(0), -1→Abnormal(1)

    records: List[Record] = []
    for wav_path in sorted(val_dir.glob("*.wav")):
        if wav_path.name.startswith("._"):
            continue
        rec_id = wav_path.stem
        label = labels.get(rec_id, -1)
        if label < 0:
            continue
        try:
            sr, data = wavfile.read(str(wav_path))
            sig = data.astype(np.float32)
            if np.abs(sig).max() > 1.0:
                sig = sig / 32768.0
            windows = preprocess_pcg_windows(sig, sr)
            if windows:
                records.append((windows, label, rec_id))
        except Exception as e:
            print(f"    WARN CinC2016val {rec_id}: {e}")
    return records


def _parse_hea_ecg(hea_path: Path) -> Tuple[Optional[np.ndarray], int, int, List[str]]:
    """Parse CinC-2021-style .hea + .mat pair.
    Returns (signal_lead_I, fs, n_samples, dx_codes).
    signal is in mV (float32).
    """
    lines = hea_path.read_text(errors="ignore").splitlines()
    # first line: recname n_leads fs n_samples
    parts = lines[0].split()
    fs = int(parts[2])
    n_samp = int(parts[3])
    # gain from first signal line (Lead I)
    gain = 1000.0
    for sig_line in lines[1:]:
        if ".mat" in sig_line and not sig_line.startswith("#"):
            tok = sig_line.split()
            # gain field like "1000.0(0)/mV"
            gain_str = tok[2].split("(")[0].split("/")[0]
            try:
                gain = float(gain_str)
            except ValueError:
                pass
            break
    # Dx codes
    dx_codes: List[str] = []
    for line in lines:
        if line.startswith("# Dx:"):
            codes_str = line.split(":", 1)[1].strip()
            dx_codes = [c.strip() for c in codes_str.split(",") if c.strip()]
            break

    mat_path = hea_path.with_suffix(".mat")
    # skip Mac resource fork ._ files
    if not mat_path.exists() or mat_path.name.startswith("._"):
        return None, fs, n_samp, dx_codes
    try:
        mat = scipy.io.loadmat(str(mat_path))
        val = mat.get("val")
        if val is None:
            return None, fs, n_samp, dx_codes
        val = np.array(val, dtype=np.float32)
        if val.ndim == 2:
            sig = val[0, :]        # Lead I
        else:
            sig = val
        sig = sig / gain           # convert ADC units → mV
        return sig, fs, n_samp, dx_codes
    except Exception as e:
        return None, fs, n_samp, dx_codes


def _dx_to_binary(dx_codes: List[str]) -> int:
    """Map SNOMED-CT Dx codes to binary label. 0=Normal, 1=Abnormal."""
    if not dx_codes:
        return -1
    # Normal only if ALL codes are in the normal set
    if all(c in _STRICT_NORMAL for c in dx_codes):
        return 0
    return 1


def load_cinc2021_ecg(base_dir: Path, subset: str) -> List[Record]:
    """Georgia / Ningbo from CinC 2021.  ECG-only, Lead I, SNOMED→binary."""
    subset_dir = base_dir / "1.0.3" / "training" / subset
    if not subset_dir.exists():
        print(f"  SKIP: {subset_dir} not found")
        return []

    records: List[Record] = []
    hea_files = sorted([
        f for f in subset_dir.rglob("*.hea")
        if not f.name.startswith("._")
    ])
    print(f"  {subset}: {len(hea_files)} .hea files found")

    skipped = 0
    for hea_path in hea_files:
        sig, fs, n_samp, dx_codes = _parse_hea_ecg(hea_path)
        if sig is None:
            skipped += 1
            continue
        label = _dx_to_binary(dx_codes)
        if label < 0:
            skipped += 1
            continue
        windows = preprocess_ecg_windows(sig, fs)
        if windows:
            records.append((windows, label, hea_path.stem))

    print(f"  {subset}: {len(records)} records loaded, {skipped} skipped")
    return records


# ════════════════════════════════════════════════════════════════════════════════
#  IN-MEMORY DATASET  (for DataLoader)
# ════════════════════════════════════════════════════════════════════════════════

class RawWindowDataset(Dataset):
    """Wraps a flat list of (window, label, modality) for batch inference."""
    def __init__(self, windows: List[np.ndarray], labels: List[int],
                 modality: str):
        self.windows = windows
        self.labels  = labels
        self.modality = modality   # "ecg" | "pcg"
        self.mel_t = 1 + (CFG.PCG_LEN - CFG.MEL_WIN) // CFG.MEL_HOP + 1

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        lbl = self.labels[i]
        if self.modality == "ecg":
            ecg_t = torch.from_numpy(w).unsqueeze(0)   # (1, ECG_LEN)
            mel_t = torch.zeros(1, CFG.MEL_N, self.mel_t)
            has_ecg = torch.tensor(1.0)
            has_pcg = torch.tensor(0.0)
        else:
            ecg_t = torch.zeros(1, CFG.ECG_LEN)
            mel_t = torch.from_numpy(w).unsqueeze(0)   # (1, MEL_N, MEL_T)
            has_ecg = torch.tensor(0.0)
            has_pcg = torch.tensor(1.0)
        return {
            "ecg":     ecg_t,
            "pcg_mel": mel_t,
            "has_ecg": has_ecg,
            "has_pcg": has_pcg,
            "label":   torch.tensor(lbl, dtype=torch.long),
        }


def _collate(batch):
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch])
    return out


# ════════════════════════════════════════════════════════════════════════════════
#  ENSEMBLE EVALUATION
# ════════════════════════════════════════════════════════════════════════════════

def flatten_records(records: List[Record], modality: str):
    """Flatten list-of-(windows, label, rec_id) into flat window/label arrays."""
    all_windows, all_labels = [], []
    for windows, label, _ in records:
        all_windows.extend(windows)
        all_labels.extend([label] * len(windows))
    return all_windows, all_labels


def evaluate_records(
    records: List[Record],
    ckpt_paths: List[str],
    device: str,
    modality: str,
    desc: str,
    batch_size: int = 128,
) -> dict:
    if not records:
        print(f"  [{desc}] SKIP: no records")
        return {"status": "empty"}

    all_windows, all_labels = flatten_records(records, modality)
    n_total = len(all_windows)
    print(f"\n  [{desc}] modality={modality}  records={len(records)}  windows={n_total}")

    # label distribution
    label_arr = np.array(all_labels)
    n_norm = (label_arr == 0).sum()
    n_abn  = (label_arr == 1).sum()
    print(f"    Normal={n_norm} ({100*n_norm/n_total:.1f}%)  Abnormal={n_abn} ({100*n_abn/n_total:.1f}%)")

    ds = RawWindowDataset(all_windows, all_labels, modality)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device == "cuda"),
                        collate_fn=_collate)

    all_probs = []
    for fold_i, ckpt in enumerate(ckpt_paths):
        model = CardioFusionSSL(enable_ssl=False, enable_recon=False).to(device)
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(sd, strict=False)
        model.eval()

        fold_probs, fold_true = [], []
        with torch.no_grad():
            for batch in loader:
                for k in ("ecg", "pcg_mel", "has_ecg", "has_pcg", "label"):
                    batch[k] = batch[k].to(device, non_blocking=True)
                with autocast("cuda", enabled=(device == "cuda" and CFG.AMP)):
                    out = model(batch, mode="supervised")
                prob = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
                fold_probs.append(prob)
                fold_true.append(batch["label"].cpu().numpy())

        all_probs.append(np.concatenate(fold_probs))
        y_true = np.concatenate(fold_true)
        del model
        torch.cuda.empty_cache()
        print(f"    fold {fold_i}: done", end="\r")

    print()
    avg_prob = np.mean(all_probs, axis=0)
    avg_pred = (avg_prob >= 0.5).astype(int)

    valid = y_true >= 0
    metrics = compute_metrics(y_true[valid], avg_pred[valid], avg_prob[valid])
    print(f"    OVERALL: N={valid.sum()}  "
          f"auroc={metrics['auroc']:.4f}  f1={metrics['f1']:.4f}  "
          f"sens={metrics['sensitivity']:.4f}  spec={metrics['specificity']:.4f}")

    return {
        "status": "ok",
        "n_samples": int(valid.sum()),
        "n_records": len(records),
        "modality": modality,
        "overall": metrics,
    }


# ════════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=str, default=None,
                    help="Directory containing fold checkpoints. Default = CFG.CHECKPOINT_DIR")
    ap.add_argument("--ckpt-glob", type=str, default="fold_*_best.pt",
                    help="Glob for checkpoint files.")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="Filename suffix — '_v2' -> external_validation_new_datasets_v2.json")
    args = ap.parse_args()

    device = CFG.device()
    print(f"[new_eval] device = {device}")

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else CFG.CHECKPOINT_DIR
    ckpt_paths = sorted(ckpt_dir.glob(args.ckpt_glob), key=lambda p: p.stem)
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints matching '{args.ckpt_glob}' in {ckpt_dir}")
    print(f"[new_eval] {len(ckpt_paths)} checkpoints found")
    ckpt_paths = [str(p) for p in ckpt_paths]

    DATASETS_ROOT = CFG.DATASETS_ROOT
    results = {}

    # ── PCG: BMD-HS ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PCG: BMD-HS (BUET multi-disease heart sound)")
    print("="*60)
    bmd_dir = DATASETS_ROOT / "bmd-hs-dataset"
    print(f"  Loading from {bmd_dir} ...")
    bmd_records = load_bmd_hs(bmd_dir)
    results["bmd_hs"] = evaluate_records(
        bmd_records, ckpt_paths, device, modality="pcg",
        desc="BMD-HS valvular disease PCG",
    )

    # ── PCG: CinC2016 validation ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PCG: CinC2016 validation set")
    print("="*60)
    val_dir = DATASETS_ROOT / "challenge-2016" / "1.0.0" / "validation"
    print(f"  Loading from {val_dir} ...")
    val_records = load_cinc2016_validation(val_dir)
    results["cinc2016_validation"] = evaluate_records(
        val_records, ckpt_paths, device, modality="pcg",
        desc="CinC2016 official validation (PCG-only)",
    )

    # ── ECG: Georgia ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ECG: Georgia 12-lead (CinC 2021)")
    print("="*60)
    cinc21_root = DATASETS_ROOT / "challenge-2021"
    georgia_records = load_cinc2021_ecg(cinc21_root, "georgia")
    results["georgia"] = evaluate_records(
        georgia_records, ckpt_paths, device, modality="ecg",
        desc="Georgia 12-lead ECG",
    )

    # ── ECG: Ningbo ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ECG: Ningbo 12-lead (CinC 2021)")
    print("="*60)
    ningbo_records = load_cinc2021_ecg(cinc21_root, "ningbo")
    results["ningbo"] = evaluate_records(
        ningbo_records, ckpt_paths, device, modality="ecg",
        desc="Ningbo 12-lead ECG",
    )

    # ── Merge with existing external_validation_results.json ─────────────────
    out_path = CFG.RESULTS_DIR / f"external_validation_results{args.out_suffix}.json"
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    existing.update(results)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\n[new_eval] Results merged -> {out_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY — New Dataset Evaluation")
    print("="*60)
    print(f"{'Dataset':<35}  {'N':>8}  {'AUROC':>7}  {'F1':>7}  "
          f"{'Sens':>7}  {'Spec':>7}  {'Modality':<8}")
    print("-" * 90)
    for name, res in results.items():
        if res.get("status") == "ok":
            m = res["overall"]
            mod = res.get("modality", "?")
            print(f"{name:<35}  {res['n_samples']:>8,}  "
                  f"{m['auroc']:>7.4f}  {m['f1']:>7.4f}  "
                  f"{m['sensitivity']:>7.4f}  {m['specificity']:>7.4f}  {mod:<8}")
        else:
            print(f"{name:<35}  SKIPPED ({res.get('status')})")
    print("\n[new_eval] DONE.")


if __name__ == "__main__":
    main()
