"""
SHAP feature attribution for the trained MambaNIDS model on TON_IoT.

Pipeline
--------
1. Re-prepare the same TON_IoT pipeline used in training.
2. Build MambaNIDS and load the saved best checkpoint.
3. Run shap.GradientExplainer on a small held-out test sample with a
   background sample drawn from the train split.
4. Aggregate the per-(time, feature, class) attribution into a single
   per-feature importance vector by taking the mean of
   |shap_value| over the time, batch, and class axes.
5. Save:
    - results/shap_top10.json (ordered top-10 features + importance)
    - Paper/fig/fig_shap_top10.png (bar plot, paper-ready)
"""
from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Force UTF-8 on Windows so any Arabic-path noise in tracebacks doesn't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import default_config, CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR     # noqa: E402
from src.data import load_ton_iot, Preprocessor, SequenceBuilder              # noqa: E402
from src.data.loader_ton_iot import CLASS_LABELS as TON_LABELS                # noqa: E402
from src.models import MambaNIDS                                              # noqa: E402

BACKGROUND_N = 100        # background samples for GradientExplainer
EXPLAIN_N    = 200        # samples to actually compute SHAP for
SEED         = 42


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def prepare_data(cfg):
    print("[1/5] loading TON_IoT")
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
    x_tr_flat, y_tr_flat = pre.fit_transform(train_df)
    x_te_flat, y_te_flat = pre.transform(test_df)

    sb = SequenceBuilder(seq_len=cfg.data.sequence_length)
    x_train, y_train = sb.transform(x_tr_flat, y_tr_flat)
    x_test, y_test = sb.transform(x_te_flat, y_te_flat)
    feature_names = list(pre.selected_features_)
    print(f"      n_features={len(feature_names)}, train={len(x_train):,}, test={len(x_test):,}")
    return (x_train, y_train), (x_test, y_test), feature_names


def main():
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = default_config()
    set_seed(SEED)

    (x_train, y_train), (x_test, y_test), feature_names = prepare_data(cfg)
    n_features = len(feature_names)

    print("[2/5] building MambaNIDS")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaNIDS(
        input_features=n_features, seq_len=cfg.data.sequence_length,
        d_model=256, n_mamba_layers=2, num_classes=cfg.model.num_classes,
    ).to(device)
    # Prefer the freshly-trained ablation 'full' checkpoint (same preprocessing
    # as today's pipeline, 30 features); fall back to the older one only if missing.
    ckpt_path = CHECKPOINT_DIR / "abl_full_best.pt"
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINT_DIR / "mamba_nids_best.pt"
    print(f"      loading {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    print("[3/5] sampling background + explain sets")
    rng = np.random.RandomState(SEED)
    bg_idx = rng.choice(len(x_train), size=min(BACKGROUND_N, len(x_train)), replace=False)
    ex_idx = rng.choice(len(x_test),  size=min(EXPLAIN_N,    len(x_test)),  replace=False)
    background = torch.tensor(x_train[bg_idx], dtype=torch.float32, device=device)
    explain    = torch.tensor(x_test[ex_idx],  dtype=torch.float32, device=device)

    print("[4/5] running shap.GradientExplainer")
    # cuDNN backward path can fail in eval mode for RNN layers; disable to be safe
    with torch.backends.cudnn.flags(enabled=False):
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(explain)

    # shap_values: list (one per class) of arrays (n_explain, T, F)
    # OR an array of shape (n_explain, T, F, n_classes) in newer SHAP versions
    sv_arr = np.array(shap_values)
    print(f"      raw shap_values shape: {sv_arr.shape}")
    # Normalise into (n_classes, n_explain, T, F) regardless of variant
    if sv_arr.ndim == 4 and sv_arr.shape[-1] == cfg.model.num_classes:
        # (n_explain, T, F, n_classes) -> (n_classes, n_explain, T, F)
        sv_arr = np.transpose(sv_arr, (3, 0, 1, 2))
    elif sv_arr.ndim == 4 and sv_arr.shape[0] == cfg.model.num_classes:
        pass  # already (n_classes, n_explain, T, F)
    elif sv_arr.ndim == 3:
        sv_arr = sv_arr[None]
    else:
        raise RuntimeError(f"unexpected shap_values shape {sv_arr.shape}")

    # Mean |shap| over class, sample, time -> (n_features,)
    importance = np.abs(sv_arr).mean(axis=(0, 1, 2))
    order = np.argsort(importance)[::-1][:10]
    top10 = [(feature_names[i], float(importance[i])) for i in order]
    print("[5/5] top-10 features by mean |SHAP|")
    for name, imp in top10:
        print(f"      {name:35s}  {imp:.5f}")

    out_json = RESULTS_DIR / "shap_top10.json"
    out_json.write_text(json.dumps({
        "model": "mamba_nids",
        "dataset": "ton_iot",
        "background_n": BACKGROUND_N,
        "explain_n": EXPLAIN_N,
        "all_features": list(feature_names),
        "all_importances": [float(v) for v in importance],
        "top10": top10,
    }, indent=2))
    print(f"      saved -> {out_json}")

    # Bar plot (paper-ready)
    fig_dir = PROJECT_ROOT.parent / "Paper" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "fig_shap_top10.png"

    names = [t[0] for t in top10][::-1]
    vals  = [t[1] for t in top10][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=160)
    ax.barh(names, vals, color="#2E5C8A", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Mean $|\\mathrm{SHAP}|$ value", fontsize=10)
    ax.set_title("MambaNIDS top-10 TON_IoT flow features by SHAP attribution",
                 fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"      saved -> {fig_path}")


if __name__ == "__main__":
    main()
