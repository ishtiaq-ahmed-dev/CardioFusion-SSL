"""Data adapter — reads CardioSense-AI .npz cache files into CardioFusion-SSL batches.

Decoupled: we do *not* import any module from CardioSense-AI; only numpy + pandas
to read the .npz + .meta.csv files that the old project's `cache_builder` already
produces.

Expected cache layout (under D:/AI_LAB_RP/CardioSense-AI/cache):
    paired_binary.npz    keys: 'ecg', 'pcg', 'binary', 'has_ecg', 'has_pcg'
    paired_binary.meta.csv  columns: subject, source, binary, ...
    pcg_binary.npz       keys: 'pcg', 'binary'
    pcg_binary.meta.csv
    ecg_binary.npz       keys: 'ecg', 'binary'
    ecg_binary.meta.csv

If your cache files have different keys, adjust ``_RESOLVE_KEYS`` below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from configs import CFG


# ---------------------------------------------------------------- cache locations
CACHE_DIR_OLD = CFG.OLD_PROJECT_ROOT / "cache"
CACHE_DIR_NEW = CFG.PROJECT_ROOT / "cache"


# ---------------------------------------------------------------- key resolution
_RESOLVE_KEYS = {
    "ecg":     ["ecg", "ecg_seg", "x_ecg"],
    "pcg":     ["pcg", "pcg_seg", "x_pcg", "pcg_wave"],
    "binary":  ["binary", "y", "label", "labels"],
    "has_ecg": ["has_ecg"],
    "has_pcg": ["has_pcg"],
}


def _first_present(npz: np.lib.npyio.NpzFile, candidates: Iterable[str]) -> Optional[str]:
    for k in candidates:
        if k in npz.files:
            return k
    return None


def load_cache(name: str, prefer_old: bool = True) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Load arrays + meta for a named cache.

    Tries CardioSense-AI/cache first, falls back to local cache/.
    Returns (arrays, meta_df).
    """
    candidates = [CACHE_DIR_OLD, CACHE_DIR_NEW] if prefer_old else [CACHE_DIR_NEW, CACHE_DIR_OLD]
    npz_path = None
    csv_path = None
    for root in candidates:
        a = root / f"{name}.npz"
        c = root / f"{name}.meta.csv"
        if a.exists() and c.exists():
            npz_path, csv_path = a, c
            break
    if npz_path is None:
        raise FileNotFoundError(
            f"No cache named {name!r} found in {[str(c) for c in candidates]}. "
            f"Run CardioSense-AI/data/cache_builder first."
        )

    with np.load(npz_path, allow_pickle=False) as npz:
        arrays = {}
        for canonical, cands in _RESOLVE_KEYS.items():
            actual = _first_present(npz, cands)
            if actual is not None:
                arrays[canonical] = np.asarray(npz[actual])

    meta = pd.read_csv(csv_path)
    return arrays, meta


# ---------------------------------------------------------------- mel-spectrogram
def _mel_spectrogram(pcg: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram from raw PCG (numpy 1-D, fs=CFG.PCG_FS).

    Uses librosa if available, else a simple torch STFT-based fallback.
    Output shape: (CFG.MEL_N, T)  where T = 1 + (PCG_LEN - MEL_WIN) // MEL_HOP + 1
    """
    try:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=pcg.astype(np.float32),
            sr=CFG.PCG_FS,
            n_fft=CFG.MEL_N_FFT,
            hop_length=CFG.MEL_HOP,
            win_length=CFG.MEL_WIN,
            n_mels=CFG.MEL_N,
            fmin=CFG.MEL_FMIN,
            fmax=CFG.MEL_FMAX,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # z-score
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        return mel_db.astype(np.float32)
    except ImportError:
        return _mel_torch_fallback(pcg)


def _mel_torch_fallback(pcg: np.ndarray) -> np.ndarray:
    """STFT-based fallback if librosa is missing. Approximate but valid."""
    x = torch.from_numpy(pcg.astype(np.float32))
    win = torch.hann_window(CFG.MEL_WIN)
    stft = torch.stft(x, n_fft=CFG.MEL_N_FFT,
                      hop_length=CFG.MEL_HOP, win_length=CFG.MEL_WIN,
                      window=win, return_complex=True, center=True)
    power = stft.abs().pow(2)               # (F, T)
    # crude mel-band aggregation: average F bins into MEL_N groups
    F_bins = power.size(0)
    grp = F_bins // CFG.MEL_N
    power = power[: grp * CFG.MEL_N].reshape(CFG.MEL_N, grp, -1).mean(dim=1)
    mel_db = 10 * torch.log10(power + 1e-8)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.numpy().astype(np.float32)


# ---------------------------------------------------------------- dataset
class PairedCacheDataset(Dataset):
    """Returns batches matching CardioFusionSSL's expected schema:

      ecg      : (1, ECG_LEN) float32
      pcg_mel  : (1, MEL_N, MEL_T) float32
      has_ecg  : scalar float 0/1
      has_pcg  : scalar float 0/1
      label    : scalar int (-1 if missing)
      subject  : str
      source   : str
    """

    def __init__(self, name: str = "paired_binary",
                 indices: Optional[np.ndarray] = None,
                 augment: bool = False):
        arrays, meta = load_cache(name)
        if "ecg" not in arrays and "pcg" not in arrays:
            raise RuntimeError(f"cache {name!r} has neither 'ecg' nor 'pcg' arrays")

        self.arrays = arrays
        self.meta = meta if indices is None else meta.iloc[indices].reset_index(drop=True)
        self.indices = indices if indices is not None else np.arange(len(meta))
        self.augment = augment
        self.has_ecg_arr = arrays.get("has_ecg")
        self.has_pcg_arr = arrays.get("has_pcg")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        orig = int(self.indices[i])
        ecg = self.arrays.get("ecg")
        pcg = self.arrays.get("pcg")

        # ECG
        if ecg is not None:
            ecg_seg = ecg[orig].astype(np.float32)
            # ensure length matches CFG.ECG_LEN
            ecg_seg = _fit_length(ecg_seg, CFG.ECG_LEN)
            has_ecg = float(self.has_ecg_arr[orig]) if self.has_ecg_arr is not None else 1.0
            ecg_t = torch.from_numpy(ecg_seg).unsqueeze(0)
        else:
            ecg_t = torch.zeros(1, CFG.ECG_LEN, dtype=torch.float32)
            has_ecg = 0.0

        # PCG -> mel
        if pcg is not None:
            pcg_seg = pcg[orig].astype(np.float32)
            pcg_seg = _fit_length(pcg_seg, CFG.PCG_LEN)
            mel = _mel_spectrogram(pcg_seg)
            mel = _fit_mel_t(mel, expected_t())
            has_pcg = float(self.has_pcg_arr[orig]) if self.has_pcg_arr is not None else 1.0
            mel_t = torch.from_numpy(mel).unsqueeze(0)
        else:
            mel_t = torch.zeros(1, CFG.MEL_N, expected_t(), dtype=torch.float32)
            has_pcg = 0.0

        row = self.meta.iloc[i]
        label = int(row["binary"]) if "binary" in row and pd.notna(row["binary"]) else -1
        subject = str(row.get("subject", f"unk_{orig}"))
        source = str(row.get("source", "unknown"))

        return {
            "ecg":     ecg_t,
            "pcg_mel": mel_t,
            "has_ecg": torch.tensor(has_ecg, dtype=torch.float32),
            "has_pcg": torch.tensor(has_pcg, dtype=torch.float32),
            "label":   torch.tensor(label, dtype=torch.long),
            "subject": subject,
            "source":  source,
        }


# ---------------------------------------------------------------- helpers
def expected_t() -> int:
    return 1 + (CFG.PCG_LEN - CFG.MEL_WIN) // CFG.MEL_HOP + 1


def _fit_length(x: np.ndarray, target: int) -> np.ndarray:
    """Pad with zeros or crop centre to exactly `target` samples."""
    n = x.shape[0]
    if n == target:
        return x
    if n > target:
        s = (n - target) // 2
        return x[s: s + target]
    pad = target - n
    left = pad // 2
    return np.pad(x, (left, pad - left), mode="constant")


def _fit_mel_t(mel: np.ndarray, target_t: int) -> np.ndarray:
    t = mel.shape[1]
    if t == target_t:
        return mel
    if t > target_t:
        s = (t - target_t) // 2
        return mel[:, s: s + target_t]
    pad = target_t - t
    left = pad // 2
    return np.pad(mel, ((0, 0), (left, pad - left)), mode="constant")


# ---------------------------------------------------------------- splits
def subject_disjoint_kfold(meta: pd.DataFrame, n_folds: int = CFG.N_FOLDS,
                           seed: int = CFG.SEED,
                           stratify_col: str = "binary"
                           ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Subject-disjoint stratified k-fold returning [(train_idx, test_idx), ...]."""
    rng = np.random.RandomState(seed)
    subj_label = (meta.groupby("subject")[stratify_col]
                  .agg(lambda s: int(s.mode().iat[0])))
    subjects = np.array(subj_label.index, dtype=object)
    labels = subj_label.values

    folds: List[List] = [[] for _ in range(n_folds)]
    for lbl in np.unique(labels):
        bucket = subjects[labels == lbl].copy()
        rng.shuffle(bucket)
        for i, subj in enumerate(bucket):
            folds[i % n_folds].append(subj)

    out = []
    for f in range(n_folds):
        test_subj = set(folds[f])
        test_idx = meta.index[meta["subject"].isin(test_subj)].to_numpy()
        train_idx = meta.index[~meta["subject"].isin(test_subj)].to_numpy()
        out.append((train_idx, test_idx))
    return out


# ---------------------------------------------------------------- dataloader builders
def build_weighted_sampler(meta: pd.DataFrame,
                           indices: np.ndarray,
                           label_col: str = "binary") -> WeightedRandomSampler:
    labels = meta.loc[indices, label_col].astype(int).values
    counts = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(weights), replacement=True)


def build_dataloader(name: str,
                     indices: Optional[np.ndarray] = None,
                     batch_size: int = CFG.BATCH_SIZE,
                     shuffle: bool = True,
                     weighted: bool = False,
                     num_workers: int = CFG.NUM_WORKERS) -> DataLoader:
    ds = PairedCacheDataset(name=name, indices=indices)
    if weighted:
        sampler = build_weighted_sampler(ds.meta, np.arange(len(ds)))
        shuffle = False
    else:
        sampler = None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=CFG.PIN_MEMORY,
        drop_last=shuffle,
    )


def collate_with_strings(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Default torch collate stacks tensors but errors on str fields. This collator
    keeps string fields as lists.
    """
    out = {}
    for k in batch[0]:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    return out
