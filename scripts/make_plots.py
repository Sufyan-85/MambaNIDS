"""
Generate all paper-style figures into results/figures/:

  fig_comparison_bars.png  — bar chart of Acc/F1/AUC across 4 models
  fig_confusion_<m>.png    — confusion matrix per model
  fig_ablation.png         — accuracy bars for ablation variants
  fig_latency.png          — latency comparison
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import default_config, CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR  # noqa: E402
from src.data import load_ton_iot, Preprocessor, SequenceBuilder           # noqa: E402
from src.data.loader_ton_iot import CLASS_LABELS as TON_LABELS              # noqa: E402
from src.training.trainer import make_loader, _resolve_device              # noqa: E402
from scripts.train import build_model                                       # noqa: E402


FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PRETTY = {
    "cnn_bilstm_mha": "CNN-BiLSTM-MHA",
    "cnn_lstm":       "CNN-LSTM",
    "gru":            "GRU",
    "lstm":           "LSTM",
}

ABLATION_PRETTY = {
    "full":           "Full",
    "no_mha":         "− MHA",
    "no_bilstm":      "− BiLSTM",
    "no_cnn":         "− CNN",
    "no_cnn_no_attn": "− CNN & Attn",
    "single_head":    "1-Head Attn",
}


# ----------------------------------------------------------------------- #
# fig 1 — comparison bar chart                                            #
# ----------------------------------------------------------------------- #
def fig_comparison(results: dict) -> None:
    models = list(MODEL_PRETTY)
    metrics_to_plot = [("accuracy", "Accuracy"), ("f1_macro", "F1 (macro)"),
                       ("auc_roc_ovr", "AUC-ROC")]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    x = np.arange(len(models))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    for i, (key, label) in enumerate(metrics_to_plot):
        vals = [results[m]["metrics"][key] * 100 for m in models]
        bars = ax.bar(x + i*width - width, vals, width, label=label, color=colors[i], edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_PRETTY[m] for m in models])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison on TON_IoT Network Dataset (test split)")
    ax.legend(loc="lower left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = FIG_DIR / "fig_comparison_bars.png"
    fig.savefig(out, dpi=180); plt.close(fig)
    print(f"  saved {out}")


# ----------------------------------------------------------------------- #
# fig 2 — latency comparison                                              #
# ----------------------------------------------------------------------- #
def fig_latency(results: dict) -> None:
    models = list(MODEL_PRETTY)
    means = [results[m]["latency_ms_mean"] for m in models]
    stds = [results[m]["latency_ms_std"] for m in models]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([MODEL_PRETTY[m] for m in models], means, yerr=stds,
                  color="#9467bd", edgecolor="black", capsize=4)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f} ms",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Inference latency (ms / batch of 512)")
    ax.set_title("Real-time inference latency")
    ax.axhline(10.0, ls="--", color="red", alpha=0.6, label="10 ms real-time threshold")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = FIG_DIR / "fig_latency.png"
    fig.savefig(out, dpi=180); plt.close(fig)
    print(f"  saved {out}")


# ----------------------------------------------------------------------- #
# fig 3 — confusion matrices                                              #
# ----------------------------------------------------------------------- #
def fig_confusion_matrices(cfg, models_to_plot: list[str]) -> None:
    # Rebuild deterministic test set
    ton_csv = DATA_DIR / "TON_IoT datasets" / "Train_Test_Network_dataset" / "train_test_network.csv"
    train_df, test_df, categorical_cols = load_ton_iot(
        ton_csv, test_ratio=cfg.data.test_ratio, random_state=cfg.data.random_state,
    )
    cfg.model.num_classes = len(TON_LABELS)

    pre = Preprocessor(
        mifs_threshold=cfg.data.mifs_threshold,
        random_state=cfg.data.random_state,
        categorical_cols=categorical_cols,
    )
    pre.fit_transform(train_df)
    x_te_flat, y_te_flat = pre.transform(test_df)
    sb = SequenceBuilder(seq_len=cfg.data.sequence_length)
    x_test, y_test = sb.transform(x_te_flat, y_te_flat)
    test_loader = make_loader(x_test, y_test, cfg.train.batch_size, False, 0)
    device = _resolve_device(cfg.train.device)

    for name in models_to_plot:
        ckpt = CHECKPOINT_DIR / f"{name}_best.pt"
        if not ckpt.exists():
            print(f"  [skip] missing checkpoint {ckpt}")
            continue
        model = build_model(name, x_test.shape[-1], cfg.data.sequence_length, cfg.model)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model = model.to(device).eval()
        preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                preds.append(model(xb.to(device)).argmax(dim=-1).cpu().numpy())
        y_pred = np.concatenate(preds)
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(TON_LABELS))))
        # Normalize by true class for readability
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        fig, ax = plt.subplots(figsize=(8, 6.5))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cm_norm[i, j]
                ax.text(j, i, f"{v*100:.1f}",
                        ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=7)
        ax.set_xticks(range(len(TON_LABELS)))
        ax.set_yticks(range(len(TON_LABELS)))
        ax.set_xticklabels(TON_LABELS, rotation=45, ha="right")
        ax.set_yticklabels(TON_LABELS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix (row-normalised %) — {MODEL_PRETTY[name]}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = FIG_DIR / f"fig_confusion_{name}.png"
        fig.savefig(out, dpi=180); plt.close(fig)
        print(f"  saved {out}")


# ----------------------------------------------------------------------- #
# fig 4 — ablation chart                                                  #
# ----------------------------------------------------------------------- #
def fig_ablation(ablation_results: list[dict]) -> None:
    order = [k for k in ABLATION_PRETTY if k in {r["variant"] for r in ablation_results}]
    variant_to_r = {r["variant"]: r for r in ablation_results}
    accs = [variant_to_r[v]["metrics"]["accuracy"] * 100 for v in order]
    f1s = [variant_to_r[v]["metrics"]["f1_macro"] * 100 for v in order]
    labels = [ABLATION_PRETTY[v] for v in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.4
    b1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#1f77b4", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width/2, f1s, width, label="F1 (macro)", color="#ff7f0e", edgecolor="black", linewidth=0.5)
    for b, v in zip(b1, accs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for b, v in zip(b2, f1s):
        ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 105); ax.set_ylabel("Score (%)")
    ax.set_title("Ablation study — contribution of each component")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = FIG_DIR / "fig_ablation.png"
    fig.savefig(out, dpi=180); plt.close(fig)
    print(f"  saved {out}")


# ----------------------------------------------------------------------- #
# main                                                                    #
# ----------------------------------------------------------------------- #
def main():
    cfg = default_config()

    results = {}
    for name in MODEL_PRETTY:
        p = RESULTS_DIR / f"{name}_results.json"
        if not p.exists():
            print(f"[warn] missing {p}; skipping that model in comparison figures")
            continue
        results[name] = json.loads(p.read_text())

    if results:
        print("Generating comparison bar chart...")
        fig_comparison(results)
        print("Generating latency chart...")
        fig_latency(results)
        print("Generating confusion matrices...")
        fig_confusion_matrices(cfg, list(results.keys()))

    abl_path = RESULTS_DIR / "ablation_results.json"
    if abl_path.exists():
        print("Generating ablation chart...")
        fig_ablation(json.loads(abl_path.read_text()))
    else:
        print("[note] ablation_results.json not found — run scripts/ablation.py first")

    print(f"\nFigures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
