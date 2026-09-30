"""
Academic-grade figures complementing the basic plots:

  fig_pr_curves.png             — per-class Precision–Recall (proposed model)
  fig_roc_overlay.png           — ROC-OVR overlay for all four models
  fig_calibration.png           — reliability diagram with ECE
  fig_per_class_f1_heatmap.png  — F1 per class × model heatmap
  fig_pareto_acc_vs_flops.png   — Accuracy vs GFLOPs Pareto frontier
  fig_pareto_acc_vs_latency.png — Accuracy vs latency Pareto
  fig_ci_forest.png             — Forest plot of 95% CIs on accuracy + F1 + κ + MCC
  fig_cd_diagram.png            — Critical Difference (Nemenyi) if Friedman available
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    roc_auc_score, roc_curve,
)
from sklearn.preprocessing import label_binarize

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

COLOURS = {
    "cnn_bilstm_mha": "#1f77b4",
    "cnn_lstm":       "#2ca02c",
    "gru":            "#d62728",
    "lstm":           "#9467bd",
}


# ----------------------------------------------------------------------- #
# data helpers                                                            #
# ----------------------------------------------------------------------- #
def _gather_probas(cfg):
    """Return {model: (y_true, y_proba)} for the TON_IoT test split."""
    train_df, test_df, categorical_cols = load_ton_iot(
        DATA_DIR / "TON_IoT datasets" / "Train_Test_Network_dataset" / "train_test_network.csv",
        test_ratio=cfg.data.test_ratio, random_state=cfg.data.random_state,
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
    n_features = x_test.shape[-1]

    out = {}
    for name in MODEL_PRETTY:
        ckpt = CHECKPOINT_DIR / f"{name}_best.pt"
        if not ckpt.exists():
            continue
        model = build_model(name, n_features, cfg.data.sequence_length, cfg.model)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model = model.to(device).eval()
        probas = []
        with torch.no_grad():
            for xb, _ in test_loader:
                p = torch.softmax(model(xb.to(device)), dim=-1).cpu().numpy()
                probas.append(p)
        out[name] = (y_test, np.concatenate(probas, axis=0))
        del model
    return out


# ----------------------------------------------------------------------- #
# Figure: per-class PR curves (proposed model only)                       #
# ----------------------------------------------------------------------- #
def fig_pr_curves(y_true, y_proba, save_to: Path) -> None:
    K = y_proba.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(K)))
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.colormaps.get_cmap("tab10")
    for c in range(K):
        if y_bin[:, c].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, c], y_proba[:, c])
        ap = average_precision_score(y_bin[:, c], y_proba[:, c])
        ax.plot(rec, prec, label=f"{TON_LABELS[c]} (AP = {ap:.3f})",
                color=cmap(c / max(1, K - 1)), linewidth=1.5)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Per-class Precision–Recall curves — CNN-BiLSTM-MHA on TON_IoT")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8, ncols=2)
    fig.tight_layout(); fig.savefig(save_to, dpi=180); plt.close(fig)
    print(f"  saved {save_to}")


# ----------------------------------------------------------------------- #
# Figure: macro-average ROC overlay for all models                         #
# ----------------------------------------------------------------------- #
def fig_roc_overlay(probas: dict, save_to: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_true, y_proba) in probas.items():
        K = y_proba.shape[1]
        y_bin = label_binarize(y_true, classes=list(range(K)))
        # macro-average TPR / FPR
        all_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(all_fpr)
        n_active = 0
        for c in range(K):
            if y_bin[:, c].sum() == 0: continue
            fpr, tpr, _ = roc_curve(y_bin[:, c], y_proba[:, c])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
            n_active += 1
        mean_tpr /= max(1, n_active)
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")
        ax.plot(all_fpr, mean_tpr, label=f"{MODEL_PRETTY[name]} (AUC = {auc:.3f})",
                color=COLOURS[name], linewidth=2.0)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Macro-averaged ROC overlay (one-vs-rest)")
    ax.grid(linestyle=":", alpha=0.5); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(save_to, dpi=180); plt.close(fig)
    print(f"  saved {save_to}")


# ----------------------------------------------------------------------- #
# Figure: calibration / reliability diagram                                #
# ----------------------------------------------------------------------- #
def fig_calibration(probas: dict, advanced: dict, save_to: Path, n_bins: int = 15) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    edges = np.linspace(0, 1, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    for name, (y_true, y_proba) in probas.items():
        conf = y_proba.max(axis=1)
        preds = y_proba.argmax(axis=1)
        accs = (preds == y_true).astype(float)
        bin_acc = np.full(n_bins, np.nan)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (conf > lo) & (conf <= hi) if i > 0 else (conf <= hi)
            if mask.any():
                bin_acc[i] = accs[mask].mean()
        ece = advanced[name]["advanced"]["ece"]
        ax.plot(centres, bin_acc, "o-",
                color=COLOURS[name], linewidth=1.8, markersize=5,
                label=f"{MODEL_PRETTY[name]} (ECE = {ece:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram (15 equal-width bins)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(linestyle=":", alpha=0.5); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(save_to, dpi=180); plt.close(fig)
    print(f"  saved {save_to}")


# ----------------------------------------------------------------------- #
# Figure: per-class F1 heatmap                                            #
# ----------------------------------------------------------------------- #
def fig_per_class_f1_heatmap(probas: dict, save_to: Path) -> None:
    K = len(TON_LABELS)
    models = list(probas.keys())
    grid = np.zeros((len(models), K))
    for i, name in enumerate(models):
        y_true, y_proba = probas[name]
        y_pred = y_proba.argmax(axis=1)
        per = f1_score(y_true, y_pred, labels=list(range(K)),
                       average=None, zero_division=0)
        grid[i, :] = per * 100

    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=40, vmax=100, aspect="auto")
    for i in range(len(models)):
        for j in range(K):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                    color="black", fontsize=8)
    ax.set_xticks(range(K)); ax.set_xticklabels(TON_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_PRETTY[m] for m in models])
    ax.set_title("Per-class F1 (%) — every model")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout(); fig.savefig(save_to, dpi=180); plt.close(fig)
    print(f"  saved {save_to}")


# ----------------------------------------------------------------------- #
# Figure: accuracy-vs-GFLOPs Pareto                                       #
# ----------------------------------------------------------------------- #
def fig_pareto(adv: dict, xkey: str, xlabel: str, save_to: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, r in adv.items():
        x = r["complexity"][xkey]
        y = r["basic"]["accuracy"] * 100
        ax.scatter(x, y, s=180, color=COLOURS[name],
                   edgecolors="black", linewidth=1.0, zorder=3)
        ax.annotate(MODEL_PRETTY[name], (x, y),
                    xytext=(8, 8), textcoords="offset points", fontsize=10)
    ax.set_xlabel(xlabel); ax.set_ylabel("Accuracy (%)")
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_title(f"Pareto frontier: Accuracy vs {xlabel}")
    fig.tight_layout(); fig.savefig(save_to, dpi=180); plt.close(fig)
    print(f"  saved {save_to}")


# ----------------------------------------------------------------------- #
# Figure: forest plot of CIs                                              #
# ----------------------------------------------------------------------- #
def fig_forest(adv: dict, save_to: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=True)
    metrics = [
        ("accuracy",  "Accuracy",   "accuracy_ci_low", "accuracy_ci_high"),
        ("f1_macro",  "F1 (macro)", "f1_macro_ci_low", "f1_macro_ci_high"),
        ("kappa",     "Cohen's κ",  "kappa_ci_low",    "kappa_ci_high"),
        ("mcc",       "MCC",        "mcc_ci_low",      "mcc_ci_high"),
    ]
    models = list(adv.keys())
    y_pos = np.arange(len(models))[::-1]
    for ax, (key, title, lo_key, hi_key) in zip(axes, metrics):
        for i, name in enumerate(models):
            if key in ("accuracy", "f1_macro"):
                point = adv[name]["basic"][key]
            else:
                point = adv[name]["advanced"][key]
            lo = adv[name]["advanced"][lo_key]
            hi = adv[name]["advanced"][hi_key]
            ax.errorbar(point, y_pos[i], xerr=[[point - lo], [hi - point]],
                        fmt="o", color=COLOURS[name], markersize=8,
                        capsize=4, elinewidth=2)
            ax.text(point, y_pos[i] + 0.15, f"{point:.3f}", ha="center", fontsize=8)
        ax.set_title(title)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([MODEL_PRETTY[m] for m in models])
        ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.suptitle("Forest plot — 95% bootstrap confidence intervals", fontsize=12)
    fig.tight_layout(); fig.savefig(save_to, dpi=180); plt.close(fig)
    print(f"  saved {save_to}")


# ----------------------------------------------------------------------- #
# Figure: Critical-Difference diagram                                     #
# ----------------------------------------------------------------------- #
def fig_cd_diagram(friedman: dict, save_to: Path) -> None:
    """Demšar-style CD diagram."""
    models = friedman["models"]
    mean_ranks = np.array(friedman["mean_ranks"])
    cd = friedman["critical_difference"]
    order = np.argsort(mean_ranks)
    models_sorted = [models[i] for i in order]
    ranks_sorted = mean_ranks[order]

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_xlim(min(ranks_sorted) - 0.5, max(ranks_sorted) + 0.5)
    ax.set_ylim(-2, 2)
    ax.axis("off")

    # Top axis
    rmin, rmax = ax.get_xlim()
    ax.plot([rmin, rmax], [0, 0], "k-", linewidth=1.5)
    for r in np.arange(np.ceil(rmin), np.floor(rmax) + 1):
        ax.plot([r, r], [-0.1, 0.1], "k-", linewidth=1)
        ax.text(r, 0.25, f"{int(r)}", ha="center", va="bottom", fontsize=9)
    ax.text((rmin + rmax) / 2, 0.55, "Mean rank (lower = better)",
            ha="center", fontsize=10, fontweight="bold")

    # Model labels above the axis
    for i, (m, r) in enumerate(zip(models_sorted, ranks_sorted)):
        side = -1 if i < len(models_sorted) / 2 else 1
        anchor_x = rmin + 0.1 if side == -1 else rmax - 0.1
        ax.plot([r, anchor_x], [0, -0.7 - 0.3 * i], "k-", linewidth=0.8)
        ax.text(anchor_x, -0.8 - 0.3 * i,
                f"{MODEL_PRETTY.get(m, m)} ({r:.2f})",
                ha="right" if side == -1 else "left", fontsize=9)

    # CD bar
    cd_x0 = rmin + 0.4
    ax.plot([cd_x0, cd_x0 + cd], [1.4, 1.4], "k-", linewidth=3)
    ax.plot([cd_x0, cd_x0], [1.3, 1.5], "k-", linewidth=2)
    ax.plot([cd_x0 + cd, cd_x0 + cd], [1.3, 1.5], "k-", linewidth=2)
    ax.text(cd_x0 + cd / 2, 1.65,
            f"CD = {cd:.2f}", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Critical-Difference diagram (Friedman + Nemenyi, α = 0.05)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(save_to, dpi=180); plt.close(fig)
    print(f"  saved {save_to}")


# ----------------------------------------------------------------------- #
# Main                                                                    #
# ----------------------------------------------------------------------- #
def main():
    cfg = default_config()
    adv_path = RESULTS_DIR / "advanced_comparison.json"
    if not adv_path.exists():
        print("Run scripts/advanced_compare.py first.")
        return 1
    adv = json.loads(adv_path.read_text())["per_model"]
    friedman = json.loads(adv_path.read_text()).get("friedman_nemenyi")

    print("Gathering predictions for plot generation...")
    probas = _gather_probas(cfg)
    proposed = probas.get("cnn_bilstm_mha")

    print("Generating advanced figures...")
    if proposed is not None:
        fig_pr_curves(proposed[0], proposed[1], FIG_DIR / "fig_pr_curves.png")
    fig_roc_overlay(probas, FIG_DIR / "fig_roc_overlay.png")
    fig_calibration(probas, adv, FIG_DIR / "fig_calibration.png")
    fig_per_class_f1_heatmap(probas, FIG_DIR / "fig_per_class_f1_heatmap.png")
    fig_pareto(adv, "gflops_per_forward", "GFLOPs / forward",
               FIG_DIR / "fig_pareto_acc_vs_flops.png")
    fig_pareto(adv, "latency_ms_mean", "Latency (ms / batch 512)",
               FIG_DIR / "fig_pareto_acc_vs_latency.png")
    fig_forest(adv, FIG_DIR / "fig_ci_forest.png")
    if friedman:
        fig_cd_diagram(friedman, FIG_DIR / "fig_cd_diagram.png")

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
