"""
Generate publication-grade block diagrams for the MambaNIDS paper:

  fig_architecture.png   — MambaNIDS architecture block diagram
  fig_pipeline.png       — end-to-end workflow pipeline
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#E8F0FE", ec="#1f4e79", lw=1.5, fs=9, bold=False):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def section_box(ax, x, y, w, h, title, fc, ec, fs=10):
    """Container with title above the inner area."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.8, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.30, title,
            ha="center", va="center",
            fontsize=fs, fontweight="bold", color=ec)


def arrow(ax, x1, y1, x2, y2, color="#333", lw=1.5):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=lw, color=color,
    )
    ax.add_patch(arr)


def label(ax, x, y, text, fs=8, color="#555"):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, color=color, style="italic")


# ======================================================================= #
# Architecture diagram                                                    #
# ======================================================================= #
def make_architecture_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 24)
    ax.axis("off")

    ax.text(5, 23.4, "MambaNIDS — End-to-End Architecture",
            ha="center", va="center", fontsize=13, fontweight="bold")

    # --- Input ---
    box(ax, 3.0, 21.6, 4.0, 0.9,
        r"Input: $\mathbf{x} \in \mathbb{R}^{T \times F}$"
        "   (T = 10 flows, F = 78 features)",
        fc="#FFE6CC", ec="#B85C00", fs=9, bold=True)
    arrow(ax, 5, 21.6, 5, 21.0)

    # --- 1D-CNN container ---
    section_box(ax, 0.5, 17.5, 9.0, 3.4,
                "1D-CNN Front End", fc="#D5E8D4", ec="#1B5E20", fs=11)
    for i, (xc, c) in enumerate(zip([2.0, 5.0, 8.0], [64, 128, 256])):
        box(ax, xc - 0.85, 18.1, 1.7, 1.4,
            f"Conv1D\n+ BN + ReLU\n+ MaxPool\n({c} filters)",
            fc="#FFFFFF", ec="#1B5E20", fs=7)
    arrow(ax, 2.85, 18.8, 4.15, 18.8, lw=1.2)
    arrow(ax, 5.85, 18.8, 7.15, 18.8, lw=1.2)
    arrow(ax, 5, 17.5, 5, 16.9)
    label(ax, 5, 17.15, "hierarchical local features")

    # --- Projection ---
    box(ax, 2.5, 15.9, 5.0, 0.8,
        r"Linear projection $\to d_{\text{model}} = 256$",
        fc="#FFF2CC", ec="#806000", fs=8)
    arrow(ax, 5, 15.9, 5, 15.3)

    # --- Mamba stack ---
    section_box(ax, 0.5, 11.4, 9.0, 3.8,
                "Mamba Temporal Stack (×2 layers)",
                fc="#E1D5E7", ec="#4B0082", fs=11)
    box(ax, 1.2, 13.4, 7.6, 1.0,
        r"RMSNorm $\to$ MambaBlock (selective scan)" "\n"
        r"$N=16$ states, $\Delta$-rank=16, expand=2, conv$_k$=4",
        fc="#FFFFFF", ec="#4B0082", fs=7)
    box(ax, 1.2, 11.9, 7.6, 1.0,
        "+ Residual connection (×2 layers)",
        fc="#FFFFFF", ec="#4B0082", fs=7)
    arrow(ax, 5, 11.4, 5, 10.8)
    label(ax, 5, 11.05, r"$\mathbf{H} \in \mathbb{R}^{T \times d}$  (long-range context)")

    # --- Forking line ---
    # Vertical down then split
    ax.plot([5, 5], [10.8, 10.3], color="#333", linewidth=1.5)
    ax.plot([2.5, 7.5], [10.3, 10.3], color="#333", linewidth=1.5)
    arrow(ax, 2.5, 10.3, 2.5, 9.8, lw=1.5)
    arrow(ax, 7.5, 10.3, 7.5, 9.8, lw=1.5)

    # --- Two attention branches ---
    box(ax, 0.4, 7.6, 4.2, 2.2,
        "Multi-Head Attention (MHA)\n"
        r"$\mathrm{Softmax}((QK^\top)/\sqrt{d_k})\,V$"
        "\nTime-domain · 8 heads · $d_k=64$",
        fc="#DAE8FC", ec="#0050B3", fs=8, bold=True)
    box(ax, 5.4, 7.6, 4.2, 2.2,
        "Frequency-Domain Attention\n"
        r"FFT $\to$ MHA on $|\widetilde{\mathbf{H}}|$ $\to$ iFFT"
        "\n(band-limited spectral path)",
        fc="#F8CECC", ec="#A02020", fs=8, bold=True)

    # Merge lines down
    arrow(ax, 2.5, 7.6, 3.5, 6.9, lw=1.5)
    arrow(ax, 7.5, 7.6, 6.5, 6.9, lw=1.5)

    # --- Gate fusion ---
    box(ax, 2.0, 5.8, 6.0, 1.1,
        "Per-token Gate Fusion\n"
        r"$\mathbf{H}^{\text{out}}_t = g_t\,\mathrm{MHA}_t + (1-g_t)\,\mathrm{FDAttn}_t$,"
        r" $\quad g_t = \sigma(\mathbf{w}_g^\top \mathbf{H}_t)$",
        fc="#FFF2CC", ec="#806000", fs=8, bold=True)
    arrow(ax, 5, 5.8, 5, 5.2)

    # --- Global pool ---
    box(ax, 2.5, 4.3, 5.0, 0.8,
        "Global Average Pool (over time)",
        fc="#E2E2E2", ec="#444", fs=9)
    arrow(ax, 5, 4.3, 5, 3.7)

    # --- FC head container ---
    section_box(ax, 0.5, 1.4, 9.0, 2.3,
                "Classification Head", fc="#FFE6CC", ec="#B85C00", fs=11)
    box(ax, 1.0, 1.7, 2.5, 0.9, "FC: 512 → 256\n+ ReLU + Dropout",
        fc="#FFFFFF", ec="#B85C00", fs=7)
    box(ax, 3.75, 1.7, 2.5, 0.9, "FC: 256 → 128\n+ ReLU + Dropout",
        fc="#FFFFFF", ec="#B85C00", fs=7)
    box(ax, 6.5, 1.7, 2.5, 0.9, "FC: 128 → C\n+ Softmax",
        fc="#FFFFFF", ec="#B85C00", fs=7)
    arrow(ax, 3.5, 2.15, 3.75, 2.15, lw=1)
    arrow(ax, 6.25, 2.15, 6.5, 2.15, lw=1)
    arrow(ax, 5, 1.4, 5, 0.95)

    # --- Output ---
    box(ax, 2.5, 0.1, 5.0, 0.8,
        r"Class probabilities $\mathbf{p} \in \Delta^{C}$ ($C=10$ attack classes)",
        fc="#D5E8D4", ec="#1B5E20", fs=9, bold=True)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ======================================================================= #
# Pipeline diagram                                                         #
# ======================================================================= #
def make_pipeline_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(14, 8.5, "End-to-End Cyberattack Detection Pipeline",
            ha="center", va="center", fontsize=13, fontweight="bold")

    # 1. Raw flows
    box(ax, 0.3, 4.5, 3.0, 2.0,
        "Raw Network\nFlows\n(Zeek / NetFlow)",
        fc="#FFE6CC", ec="#B85C00", fs=10, bold=True)
    arrow(ax, 3.3, 5.5, 4.0, 5.5)

    # 2. Preprocessing (taller box with title at top)
    section_box(ax, 4.0, 3.0, 4.5, 4.0,
                "Preprocessing", fc="#D5E8D4", ec="#1B5E20", fs=11)
    box(ax, 4.3, 5.7, 3.9, 0.6,
        "1. Median imputation",
        fc="#FFFFFF", ec="#1B5E20", fs=7)
    box(ax, 4.3, 5.0, 3.9, 0.6,
        r"2. 99.5$^{\rm th}$-pct outlier clip",
        fc="#FFFFFF", ec="#1B5E20", fs=7)
    box(ax, 4.3, 4.3, 3.9, 0.6,
        r"3. Min--max norm to $[0,1]$",
        fc="#FFFFFF", ec="#1B5E20", fs=7)
    box(ax, 4.3, 3.6, 3.9, 0.6,
        r"4. MIFS ($\geq 0.05$)",
        fc="#FFFFFF", ec="#1B5E20", fs=7)
    arrow(ax, 8.5, 5.5, 9.2, 5.5)

    # 3. Sliding windows
    box(ax, 9.2, 4.5, 3.0, 2.0,
        "Sliding\nWindows\n(T = 10)",
        fc="#DAE8FC", ec="#0050B3", fs=10, bold=True)
    arrow(ax, 12.2, 5.5, 12.9, 5.5)

    # 4. SMOTE
    box(ax, 12.9, 4.5, 3.2, 2.0,
        "SMOTE\nOversampling\n(training only)",
        fc="#FFF2CC", ec="#806000", fs=10, bold=True)
    arrow(ax, 16.1, 5.5, 16.8, 5.5)

    # 5. MambaNIDS
    section_box(ax, 16.8, 3.0, 5.4, 4.0,
                "MambaNIDS", fc="#E1D5E7", ec="#4B0082", fs=12)
    box(ax, 17.1, 5.7, 4.8, 0.6,
        r"1D-CNN (3 blocks, 64$\to$128$\to$256)",
        fc="#FFFFFF", ec="#4B0082", fs=7)
    box(ax, 17.1, 5.0, 4.8, 0.6,
        r"Mamba stack ($\times 2$, $N=16$)",
        fc="#FFFFFF", ec="#4B0082", fs=7)
    box(ax, 17.1, 4.3, 4.8, 0.6,
        "Dual MHA + FDAttn (gated)",
        fc="#FFFFFF", ec="#4B0082", fs=7)
    box(ax, 17.1, 3.6, 4.8, 0.6,
        r"FC head (512$\to$256$\to$128$\to$C)",
        fc="#FFFFFF", ec="#4B0082", fs=7)
    arrow(ax, 22.2, 5.5, 22.9, 5.5)

    # 6. Output
    section_box(ax, 22.9, 3.0, 4.8, 4.0,
                "Prediction", fc="#F8CECC", ec="#A02020", fs=11)
    box(ax, 23.1, 5.7, 4.4, 0.6,
        "Normal traffic",
        fc="#D5E8D4", ec="#A02020", fs=8)
    box(ax, 23.1, 5.0, 4.4, 0.6,
        "Attack class (DoS, DDoS, ...)",
        fc="#FFE6CC", ec="#A02020", fs=8)
    box(ax, 23.1, 4.3, 4.4, 0.6,
        r"Confidence + Cohen's $\kappa$",
        fc="#FFFFFF", ec="#A02020", fs=8)
    box(ax, 23.1, 3.6, 4.4, 0.6,
        "Adversarial-robust score",
        fc="#FFFFFF", ec="#A02020", fs=8)

    # Annotations at bottom
    ax.text(14, 1.9,
            r"Train / Validation / Test = 70 / 15 / 15 (stratified, seeds $\{42, 7, 123\}$)",
            ha="center", va="center", fontsize=9, color="#444", style="italic")
    ax.text(14, 1.0,
            r"Loss: Cross-entropy + $\ell_2(\lambda=10^{-4})$ "
            r"$\,\bullet\,$ Adam ($\eta=10^{-3}$, cosine annealing) "
            r"$\,\bullet\,$ Batch 512  $\,\bullet\,$ 15 epochs",
            ha="center", va="center", fontsize=8, color="#444", style="italic")

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


if __name__ == "__main__":
    print("Generating block diagrams...")
    make_architecture_diagram(FIG_DIR / "fig_architecture.png")
    make_pipeline_diagram(FIG_DIR / "fig_pipeline.png")
    print(f"\nAll diagrams saved to {FIG_DIR}")
