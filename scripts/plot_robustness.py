"""Plot FGSM and PGD robustness curves from results/robustness.json."""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PRETTY = {
    "mamba_nids":      "MambaNIDS (proposed)",
    "cnn_bilstm_mha":  "CNN-BiLSTM-MHA",
    "transformer_ids": "Transformer-IDS",
    "cnn_lstm":        "CNN-LSTM",
    "lstm":            "LSTM",
    "gru":             "GRU",
}
COLOURS = {
    "mamba_nids":      "#1f77b4",
    "cnn_bilstm_mha":  "#ff7f0e",
    "transformer_ids": "#2ca02c",
    "cnn_lstm":        "#d62728",
    "lstm":            "#9467bd",
    "gru":             "#8c564b",
}
MARKERS = {
    "mamba_nids":      "o",
    "cnn_bilstm_mha":  "s",
    "transformer_ids": "D",
    "cnn_lstm":        "^",
    "lstm":            "v",
    "gru":             "P",
}


def plot_attack(d, attack, out_path):
    epss = d["epsilons"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for m in d["models"]:
        name = m["model"]
        accs = [a * 100 for a in m[attack]["accuracies"]]
        ax.plot(epss, accs,
                marker=MARKERS.get(name, "o"),
                color=COLOURS.get(name, "black"),
                linewidth=2.0, markersize=8,
                label=MODEL_PRETTY.get(name, name))
    ax.set_xlabel(r"Perturbation budget $\epsilon$ ($\ell_\infty$)")
    ax.set_ylabel("Accuracy (%)")
    title = "FGSM" if attack == "fgsm" else f"PGD-{d.get('pgd_steps', 10)}"
    ax.set_title(f"Adversarial robustness under {title} attack")
    ax.set_xlim(min(epss) - 0.005, max(epss) + 0.005)
    ax.set_ylim(0, 100)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180); plt.close(fig)
    print(f"saved {out_path}")


def plot_both(d, out_path):
    """Side-by-side FGSM + PGD on one figure."""
    epss = d["epsilons"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, attack, title in zip(axes, ("fgsm", "pgd"),
                                  ("FGSM", f"PGD-{d.get('pgd_steps', 10)}")):
        for m in d["models"]:
            name = m["model"]
            accs = [a * 100 for a in m[attack]["accuracies"]]
            ax.plot(epss, accs,
                    marker=MARKERS.get(name, "o"),
                    color=COLOURS.get(name, "black"),
                    linewidth=2.0, markersize=7,
                    label=MODEL_PRETTY.get(name, name))
        ax.set_xlabel(r"Perturbation budget $\epsilon$ ($\ell_\infty$)")
        ax.set_title(title)
        ax.set_xlim(min(epss) - 0.005, max(epss) + 0.005)
        ax.set_ylim(0, 100)
        ax.grid(linestyle=":", alpha=0.5)
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].legend(loc="upper right", fontsize=9)
    fig.suptitle("Adversarial robustness — TON_IoT test set", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180); plt.close(fig)
    print(f"saved {out_path}")


def main():
    d = json.loads((ROOT / "results" / "robustness.json").read_text())
    plot_attack(d, "fgsm", FIG_DIR / "fig_robustness_fgsm.png")
    plot_attack(d, "pgd",  FIG_DIR / "fig_robustness_pgd.png")
    plot_both(d,           FIG_DIR / "fig_robustness_combined.png")


if __name__ == "__main__":
    main()
