"""Paper-grade visualisations for CardioFusion-SSL results.

All figures saved as both PNG (300 DPI) and PDF. Designed to be directly
usable in the manuscript without modification.

Functions:
    plot_roc_curves          — per-fold + mean ROC, with shaded CI
    plot_confusion_matrix    — heatmap (absolute counts + normalised)
    plot_training_curves     — SSL loss/top1 and supervised loss/acc over epochs
    plot_cross_dataset_bars  — grouped bar chart of accuracy/F1/AUROC per dataset
    plot_ablation_bars       — horizontal bar chart of ablation deltas
    plot_attention_heatmap   — cross-modal attention weights at each fusion scale
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for background rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import roc_curve, auc

from configs import CFG

OUT = CFG.RESULTS_DIR / "plots"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "ecg":    "#1f77b4",
    "pcg":    "#d62728",
    "fusion": "#2ca02c",
    "ssl":    "#9467bd",
    "mean":   "#2ca02c",
    "ci":     "#a8d5a2",
}
GRID_KW  = dict(alpha=0.3, linestyle="--")
SAVE_KW  = dict(dpi=CFG.PLOT_DPI, bbox_inches="tight")


def _savefig(name: str, fig: plt.Figure) -> None:
    for fmt in CFG.PLOT_FMT:
        p = OUT / f"{name}.{fmt}"
        fig.savefig(p, **SAVE_KW)
    plt.close(fig)
    print(f"  [vis] saved {OUT / name}.png")


# --------------------------------------------------------------------- ROC curves
def plot_roc_curves(fold_results: list[dict], title: str = "10-Fold Cross-Validation ROC",
                    name: str = "roc_kfold") -> None:
    """Plot per-fold ROC curves with mean ± std envelope."""
    fig, ax = plt.subplots(figsize=(6, 6))
    tprs, aucs = [], []
    base_fpr = np.linspace(0, 1, 101)

    for r in fold_results:
        y_true = np.array(r["y_true"])
        y_prob = np.array(r["y_prob"])
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        interp_tpr = np.interp(base_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        ax.plot(base_fpr, interp_tpr, alpha=0.3, color=COLORS["fusion"], lw=1)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(base_fpr, mean_tpr)
    std_tpr = np.std(tprs, axis=0)

    ax.plot(base_fpr, mean_tpr, color=COLORS["mean"], lw=2,
            label=f"Mean ROC (AUC = {mean_auc:.3f} ± {np.std(aucs):.3f})")
    ax.fill_between(base_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    color=COLORS["ci"], alpha=0.4, label="± 1 std dev")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(**GRID_KW)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    _savefig(name, fig)


# --------------------------------------------------------------------- confusion matrix
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          classes: tuple = CFG.BINARY_CLASSES,
                          name: str = "confusion_matrix") -> None:
    from sklearn.metrics import confusion_matrix as sk_cm
    cm = sk_cm(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, fmt, ttl in zip(
            axes, [cm, cm_norm], ["d", ".2f"],
            ["Counts", "Normalised (row-wise)"]):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax, shrink=0.8)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks); ax.set_xticklabels(classes, fontsize=11)
        ax.set_yticks(tick_marks); ax.set_yticklabels(classes, fontsize=11)
        thresh = data.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center", fontsize=13,
                        color="white" if data[i, j] > thresh else "black")
        ax.set_ylabel("True label", fontsize=11)
        ax.set_xlabel("Predicted label", fontsize=11)
        ax.set_title(ttl, fontsize=12)
    fig.suptitle("Confusion Matrix — CardioFusion-SSL", fontsize=13, y=1.01)
    fig.tight_layout()
    _savefig(name, fig)


# --------------------------------------------------------------------- training curves
def plot_training_curves(ssl_history_path: Optional[str] = None,
                         finetune_results_path: Optional[str] = None,
                         name: str = "training_curves") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # SSL panel
    ax = axes[0]
    if ssl_history_path and Path(ssl_history_path).exists():
        hist = json.load(open(ssl_history_path))
        epochs = [h["epoch"] for h in hist]
        losses = [h["loss"] for h in hist]
        top1s  = [h["top1"] for h in hist]
        ax.plot(epochs, losses, color=COLORS["ssl"], lw=2, label="InfoNCE loss")
        ax2 = ax.twinx()
        ax2.plot(epochs, top1s, color=COLORS["fusion"], lw=2, linestyle="--",
                 label="Top-1 retrieval")
        ax2.set_ylabel("Top-1 Retrieval Accuracy", fontsize=11, color=COLORS["fusion"])
        ax2.tick_params(axis="y", colors=COLORS["fusion"])
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss", fontsize=11)
        ax.set_title("SSL Pretraining", fontsize=12)
        ax.grid(**GRID_KW)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    else:
        ax.text(0.5, 0.5, "SSL history not available", ha="center", va="center",
                transform=ax.transAxes)

    # Finetune panel — per-fold AUROC
    ax = axes[1]
    if finetune_results_path and Path(finetune_results_path).exists():
        data = json.load(open(finetune_results_path))
        folds = [r["fold"] for r in data["per_fold"]]
        aucs  = [r["metrics"]["auroc"] for r in data["per_fold"]]
        f1s   = [r["metrics"]["f1"] for r in data["per_fold"]]
        x = np.arange(len(folds))
        w = 0.35
        ax.bar(x - w/2, aucs, w, color=COLORS["fusion"], alpha=0.8, label="AUROC")
        ax.bar(x + w/2, f1s,  w, color=COLORS["ecg"],   alpha=0.8, label="F1")
        ax.axhline(np.mean(aucs), color=COLORS["fusion"], linestyle="--", lw=1, alpha=0.6)
        ax.axhline(np.mean(f1s),  color=COLORS["ecg"],   linestyle="--", lw=1, alpha=0.6)
        ax.set_xlabel("Fold"); ax.set_ylabel("Score")
        ax.set_title("Supervised Fine-Tuning — Per-Fold AUROC & F1", fontsize=12)
        ax.set_xticks(x); ax.set_xticklabels([f"F{f}" for f in folds])
        ax.legend(fontsize=10); ax.grid(**GRID_KW)
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "Fine-tune results not available", ha="center", va="center",
                transform=ax.transAxes)

    fig.suptitle("CardioFusion-SSL Training Summary", fontsize=13)
    fig.tight_layout()
    _savefig(name, fig)


# --------------------------------------------------------------------- cross-dataset bar chart
# Human-readable labels for each dataset key
_DS_LABELS = {
    "circor":               "CirCor\n(PCG, paed.)",
    "cinc2016_external":    "CinC2016\nb-f (PCG)",
    "cinc2016_validation":  "CinC2016\nval (PCG)",
    "bmd_hs":               "BMD-HS\n(PCG valv.)",
    "chapman":              "Chapman\n(ECG)",
    "ptbxl":                "PTB-XL\n(ECG)",
    "mitbih":               "MIT-BIH\n(ECG)",
    "cpsc2018":             "CPSC-2018\n(ECG)",
    "georgia":              "Georgia\n(ECG)",
    "ningbo":               "Ningbo\n(ECG)",
}
# Group modality colours
_DS_MOD = {
    "circor": "pcg", "cinc2016_external": "pcg",
    "cinc2016_validation": "pcg", "bmd_hs": "pcg",
    "chapman": "ecg", "ptbxl": "ecg", "mitbih": "ecg",
    "cpsc2018": "ecg", "georgia": "ecg", "ningbo": "ecg",
}


def plot_cross_dataset_bars(eval_results_path: str,
                            metrics: list = ("f1", "auroc", "sensitivity"),
                            name: str = "cross_dataset_eval") -> None:
    """Bar chart of external validation metrics across all datasets.

    Handles both the legacy nested format {ensemble: {ds: {overall: {...}}}}
    and the flat format {ds: {overall: {...}}} produced by run_external_eval.py
    and eval_new_datasets.py.
    """
    data = json.load(open(eval_results_path))

    # detect flat vs. nested format
    first_val = next(iter(data.values()), {})
    if "overall" in first_val or first_val.get("status") == "ok":
        ds_dict = data   # flat format
    else:
        target = "ensemble" if "ensemble" in data else list(data.keys())[0]
        ds_dict = data.get(target, data)

    datasets, rows = [], {m: [] for m in metrics}
    for ds_name, res in ds_dict.items():
        if res.get("status") != "ok":
            continue
        overall = res.get("overall", {})
        # skip datasets where ALL requested metrics are NaN
        vals = [overall.get(m, float("nan")) for m in metrics]
        if all(v != v for v in vals):   # all nan
            continue
        datasets.append(ds_name)
        for m in metrics:
            v = overall.get(m, float("nan"))
            rows[m].append(v if (v == v) else 0.0)   # replace nan with 0

    if not datasets:
        print("  [vis] no valid cross-dataset results to plot")
        return

    # order: PCG first, ECG second
    pcg_ds = [d for d in datasets if _DS_MOD.get(d) == "pcg"]
    ecg_ds = [d for d in datasets if _DS_MOD.get(d) == "ecg"]
    other   = [d for d in datasets if d not in pcg_ds and d not in ecg_ds]
    ordered = pcg_ds + ecg_ds + other
    # reorder rows accordingly
    idx_map = {d: i for i, d in enumerate(datasets)}
    for m in metrics:
        rows[m] = [rows[m][idx_map[d]] for d in ordered]
    datasets = ordered

    x = np.arange(len(datasets))
    w = 0.8 / len(metrics)
    met_cols = ["#2ca02c", "#1f77b4", "#d62728", "#9467bd"]   # f1, auroc, sens, spec
    fig, ax = plt.subplots(figsize=(max(12, 1.6 * len(datasets)), 5))

    for i, m in enumerate(metrics):
        offset = (i - len(metrics) / 2 + 0.5) * w
        bar_colors = [
            "#d62728" if _DS_MOD.get(d) == "pcg" else "#1f77b4"
            for d in datasets
        ]
        bars = ax.bar(x + offset, rows[m], w, label=m.upper(),
                      color=met_cols[i % len(met_cols)], alpha=0.82, edgecolor="white")
        for bar, val in zip(bars, rows[m]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    # Divider between PCG / ECG groups
    if pcg_ds and ecg_ds:
        ax.axvline(len(pcg_ds) - 0.5, color="gray", linestyle=":", lw=1.5, alpha=0.7)
        ax.text(len(pcg_ds) / 2 - 0.5, 1.05, "PCG-only mode",
                ha="center", fontsize=9, color="#d62728", transform=ax.get_xaxis_transform())
        ax.text(len(pcg_ds) + len(ecg_ds) / 2 - 0.5, 1.05, "ECG-only mode",
                ha="center", fontsize=9, color="#1f77b4", transform=ax.get_xaxis_transform())

    ax.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.5, label="Chance (0.5)")
    ax.set_xlabel("External Dataset", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("CardioFusion-SSL — Cross-Dataset External Validation (10-Fold Ensemble)", fontsize=12)
    labels = [_DS_LABELS.get(d, d) for d in datasets]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
    ax.legend(fontsize=9, loc="upper right"); ax.grid(**GRID_KW)
    ax.set_ylim(0, 1.18)
    fig.tight_layout()
    _savefig(name, fig)


# --------------------------------------------------------------------- ablation bar chart
def plot_ablation_bars(ablation_results: dict,
                       baseline_metric: str = "auroc",
                       name: str = "ablation_study") -> None:
    """
    ablation_results: {variant_name: {metric: value}}
    e.g. {"full_model": {...}, "no_ssl": {...}, "single_scale": {...}, ...}
    """
    full = ablation_results.get("full_model", {})
    baseline = full.get(baseline_metric, 1.0)
    variants = [k for k in ablation_results if k != "full_model"]
    deltas = [ablation_results[v].get(baseline_metric, 0.0) - baseline for v in variants]
    colors = ["#e74c3c" if d < 0 else "#2ca02c" for d in deltas]

    fig, ax = plt.subplots(figsize=(8, max(3, len(variants) * 0.6)))
    bars = ax.barh(variants, deltas, color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=1.2)
    for bar, d in zip(bars, deltas):
        ax.text(d + (0.001 if d >= 0 else -0.001), bar.get_y() + bar.get_height() / 2,
                f"{d:+.4f}", va="center", ha="left" if d >= 0 else "right", fontsize=9)
    ax.set_xlabel(f"Δ {baseline_metric.upper()} vs. Full Model", fontsize=11)
    ax.set_title(f"Ablation Study — Impact on {baseline_metric.upper()}", fontsize=12)
    ax.grid(**GRID_KW)
    fig.tight_layout()
    _savefig(name, fig)


# --------------------------------------------------------------------- reliability diagram
def plot_reliability_diagram(stats_results_path: str,
                              name: str = "reliability_diagram") -> None:
    """Plot ECE reliability diagram from statistical_tests.json calibration bins."""
    data = json.load(open(stats_results_path))
    ece_data = data.get("ece", {})
    bins = ece_data.get("bins", [])
    ece  = ece_data.get("ece", float("nan"))

    if not bins:
        print("  [vis] no calibration bins in stats results")
        return

    conf_vals = np.array([b["conf"] for b in bins])
    acc_vals  = np.array([b["acc"] for b in bins])
    n_vals    = np.array([b["n"] for b in bins])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Reliability diagram
    ax = axes[0]
    mask = n_vals > 0
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    ax.bar(conf_vals[mask], acc_vals[mask], width=0.09,
           color=COLORS["fusion"], alpha=0.7, label="Model")
    ax.set_xlabel("Mean confidence", fontsize=11)
    ax.set_ylabel("Fraction of positives", fontsize=11)
    ax.set_title(f"Reliability Diagram  (ECE = {ece:.4f})", fontsize=12)
    ax.legend(fontsize=10); ax.grid(**GRID_KW)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)

    # Histogram of confidence values
    ax = axes[1]
    ax.bar(conf_vals[mask], n_vals[mask], width=0.09,
           color=COLORS["ecg"], alpha=0.7)
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_ylabel("N windows", fontsize=11)
    ax.set_title("Confidence distribution", fontsize=12)
    ax.grid(**GRID_KW)

    fig.tight_layout()
    _savefig(name, fig)


# --------------------------------------------------------------------- main (generate all from disk)
def generate_all() -> None:
    """Generate every available plot from existing result files."""
    ssl_hist  = str(CFG.RESULTS_DIR / "ssl_pretrain_history.json")
    ft_res    = str(CFG.RESULTS_DIR / "finetune_results.json")
    ext_res   = str(CFG.RESULTS_DIR / "external_validation.json")
    abl_res   = str(CFG.RESULTS_DIR / "ablation_results.json")

    print("[vis] Generating training curves ...")
    plot_training_curves(ssl_hist, ft_res)

    if Path(ft_res).exists():
        data = json.load(open(ft_res))
        print("[vis] Generating ROC curves ...")
        plot_roc_curves(data["per_fold"])

        # aggregate confusion matrix across all folds
        y_true_all = np.concatenate([r["y_true"] for r in data["per_fold"]])
        y_pred_all = np.concatenate([r["y_pred"] for r in data["per_fold"]])
        print("[vis] Generating confusion matrix ...")
        plot_confusion_matrix(y_true_all, y_pred_all)

    if Path(ext_res).exists():
        print("[vis] Generating cross-dataset bars ...")
        plot_cross_dataset_bars(ext_res)

    if Path(abl_res).exists():
        print("[vis] Generating ablation bars ...")
        plot_ablation_bars(json.load(open(abl_res)))

    print("[vis] All plots saved to", OUT)


if __name__ == "__main__":
    generate_all()
