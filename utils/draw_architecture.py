"""Generate the CardioFusion-SSL architecture diagram (Figure 1 of paper).

Run standalone: python utils/draw_architecture.py
Saves to results/plots/architecture_diagram.{png,pdf}
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)

# ── colour palette ──────────────────────────────────────────────────
C_ECG    = "#1f77b4"   # blue
C_PCG    = "#d62728"   # red
C_FUSE   = "#2ca02c"   # green
C_SSL    = "#9467bd"   # purple
C_HEAD   = "#8c564b"   # brown
C_ARROW  = "#555555"
C_BG     = "#f9f9f9"
C_BOX    = "#ffffff"
ALPHA_FILL = 0.15

def box(ax, x, y, w, h, color, label, sublabel=None, fontsize=9, alpha=ALPHA_FILL):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.04",
                           linewidth=1.5, edgecolor=color,
                           facecolor=color, alpha=alpha)
    ax.add_patch(rect)
    ax.text(x, y + (0.07 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize, fontweight="bold", color=color)
    if sublabel:
        ax.text(x, y - 0.08, sublabel,
                ha="center", va="center", fontsize=7, color="#444444")

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, style="->", lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def dashed_box(ax, x, y, w, h, color, label, fontsize=8):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.04",
                           linewidth=1.5, edgecolor=color, linestyle="--",
                           facecolor=color, alpha=0.07)
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            color=color, style="italic")

# ── figure ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(-0.5, 14.5)
ax.set_ylim(-0.5, 8.5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_facecolor(C_BG)
fig.patch.set_facecolor(C_BG)

# ── title ────────────────────────────────────────────────────────────
ax.text(7, 8.1, "CardioFusion-SSL Architecture", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#222222")

# ── inputs ──────────────────────────────────────────────────────────
box(ax, 1.5, 6.5, 2.2, 0.55, C_ECG, "ECG Input", "(B, 1, 2000) @ 500 Hz")
box(ax, 1.5, 1.5, 2.2, 0.55, C_PCG, "PCG Input", "(B, 1, 128, 111) mel")

# ── ECG encoder ─────────────────────────────────────────────────────
box(ax, 4.0, 7.2, 2.2, 0.5, C_ECG, "Patch Stem", "Conv1d k=25 s=25 → 80 tok")
box(ax, 4.0, 6.5, 2.2, 0.5, C_ECG, "Pos. Embed", "learnable sinusoidal")
box(ax, 4.0, 5.7, 2.2, 0.75, C_ECG, "ECG Encoder", "6× Transformer blocks\n(Mamba optional)\n→ (B, 80, 256)")

# draw ECG encoder brace
dashed_box(ax, 4.0, 6.47, 2.5, 2.2, C_ECG, "ECG Encoder (4.7 M params)")

# ── PCG encoder ─────────────────────────────────────────────────────
box(ax, 4.0, 2.2, 2.2, 0.5, C_PCG, "2D Patch Embed", "Conv2d (16×16) → 48 tok")
box(ax, 4.0, 1.5, 2.2, 0.5, C_PCG, "Pos. Embed", "learnable 2D position")
box(ax, 4.0, 0.7, 2.2, 0.75, C_PCG, "PCG Encoder", "6× AST blocks\n→ (B, 49, 256)")

dashed_box(ax, 4.0, 1.47, 2.5, 2.2, C_PCG, "PCG Encoder (4.8 M params)")

# ── arrows to encoders ──────────────────────────────────────────────
arrow(ax, 2.6, 6.5, 2.9, 6.5, C_ECG)
arrow(ax, 2.6, 1.5, 2.9, 1.5, C_PCG)

# ── fusion module ────────────────────────────────────────────────────
# three scale boxes
for i, (s_name, s_tok, y_fuse) in enumerate([
        ("Fine scale\n(s=4, ~250ms/tok)", "4 tok", 7.0),
        ("Mid scale\n(s=16, ~62ms/tok)",  "16 tok", 4.0),
        ("Coarse scale\n(s=64, ~16ms/tok)", "64 tok", 1.0)]):
    box(ax, 8.0, y_fuse, 2.5, 1.3, C_FUSE, f"BiCrossAttn ×2", s_name, fontsize=8)
    ax.text(8.0, y_fuse - 0.55, s_tok, ha="center", va="center",
            fontsize=7, color=C_FUSE, style="italic")

dashed_box(ax, 8.0, 4.0, 3.0, 7.0, C_FUSE, "Hierarchical Fusion (9.5 M params)")
ax.text(8.0, 7.7, "Adaptive Pool ↕", ha="center", fontsize=7.5, color=C_FUSE)
ax.text(8.0, 0.3, "⊕ Concat → (B, 1536)", ha="center", fontsize=7.5, color=C_FUSE,
        fontweight="bold")

# arrows into fusion from ECG encoder
arrow(ax, 5.1, 5.7, 6.75, 6.8, C_ECG)
arrow(ax, 5.1, 5.7, 6.75, 3.85, C_ECG)
arrow(ax, 5.1, 5.7, 6.75, 0.85, C_ECG)

# arrows into fusion from PCG encoder
arrow(ax, 5.1, 0.7, 6.75, 1.15, C_PCG)
arrow(ax, 5.1, 0.7, 6.75, 4.15, C_PCG)
arrow(ax, 5.1, 0.7, 6.75, 7.15, C_PCG)

# ── missing-modality tokens ──────────────────────────────────────────
box(ax, 6.3, 5.2, 1.7, 0.45, "#ff7f0e", "Miss-ECG tok", "(1,1,256) learned")
box(ax, 6.3, 2.8, 1.7, 0.45, "#ff7f0e", "Miss-PCG tok", "(1,1,256) learned")
ax.text(6.3, 4.7, "↑ when has_ecg=0", ha="center", fontsize=6.5, color="#ff7f0e")
ax.text(6.3, 3.3, "↑ when has_pcg=0", ha="center", fontsize=6.5, color="#ff7f0e")

# ── classifier head ──────────────────────────────────────────────────
box(ax, 11.0, 4.0, 2.2, 0.9, C_HEAD, "Classifier Head",
    "Linear(1536→512)\nLayerNorm→GELU→Drop\nLinear(512→2)")

arrow(ax, 9.25, 4.0, 9.89, 4.0, C_FUSE, lw=2)

# ── output ───────────────────────────────────────────────────────────
box(ax, 13.2, 4.0, 1.8, 0.5, "#333333", "Normal / Abnormal",
    "softmax prob", fontsize=8)
arrow(ax, 12.1, 4.0, 12.3, 4.0, C_HEAD, lw=2)

# ── SSL branch (dashed, above) ───────────────────────────────────────
box(ax, 6.3, 8.0, 2.0, 0.45, C_SSL, "Proj Head (ECG)", "MLP 256→128 L2")
box(ax, 9.0, 8.0, 2.0, 0.45, C_SSL, "Proj Head (PCG)", "MLP 256→128 L2")
box(ax, 7.65, 7.15, 2.2, 0.45, C_SSL, "InfoNCE Loss", "temp τ=0.07")

arrow(ax, 4.0, 6.3, 5.3, 7.95, C_SSL, style="->")
arrow(ax, 7.25, 8.0, 8.0, 8.0, C_SSL)
arrow(ax, 6.3, 7.78, 6.9, 7.35, C_SSL)
arrow(ax, 9.0, 7.78, 8.4, 7.35, C_SSL)
# note: PCG proj head
arrow(ax, 4.0, 1.7, 5.3, 7.85, C_SSL, style="->")

ax.text(7.65, 7.55, "SSL pretraining only", ha="center", fontsize=7,
        color=C_SSL, style="italic")

# ── legend ───────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_ECG, label="ECG encoder"),
    mpatches.Patch(color=C_PCG, label="PCG encoder"),
    mpatches.Patch(color=C_FUSE, label="Hierarchical fusion"),
    mpatches.Patch(color=C_SSL, label="SSL pretraining (not used at inference)"),
    mpatches.Patch(color="#ff7f0e", label="Missing-modality tokens"),
    mpatches.Patch(color=C_HEAD, label="Classifier head"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=8,
          bbox_to_anchor=(14.4, 0.0), framealpha=0.9)

# ── param count annotation ────────────────────────────────────────────
ax.text(0.0, 0.0, "Total: 20.5 M parameters",
        ha="left", va="bottom", fontsize=9, color="#444444",
        transform=ax.transData)

fig.tight_layout(pad=0.3)
for fmt in ("png", "pdf"):
    fig.savefig(OUT / f"architecture_diagram.{fmt}", dpi=300, bbox_inches="tight",
                facecolor=C_BG)
    print(f"  [arch] saved -> {OUT}/architecture_diagram.{fmt}")
plt.close(fig)
print("[arch] DONE")
