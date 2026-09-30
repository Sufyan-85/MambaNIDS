"""
Ablation study — mirrors Table 5 of the paper.

Runs the following six configurations on TON_IoT and writes their metrics
to results/ablation_results.json:

  1. Full CNN-BiLSTM-MHA            (baseline of the table)
  2. w/o Multi-Head Attention
  3. w/o BiLSTM     (use unidirectional LSTM instead)
  4. w/o CNN        (BiLSTM-MHA only on raw features)
  5. w/o CNN & Attention
  6. Single-head Attention (n_heads=1)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import default_config, CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR  # noqa: E402
from src.data import load_ton_iot, Preprocessor, SequenceBuilder           # noqa: E402
from src.data.loader_ton_iot import CLASS_LABELS as TON_LABELS              # noqa: E402
from src.models import CNNBiLSTMMHA                                         # noqa: E402
from src.training import Trainer                                            # noqa: E402
from src.training.trainer import make_loader                                # noqa: E402
from src.evaluation import compute_metrics, latency_benchmark               # noqa: E402


ABLATION_CONFIGS = {
    "full":          {"use_cnn": True,  "use_bilstm": True,  "use_attention": True,  "n_heads": 8},
    "no_mha":        {"use_cnn": True,  "use_bilstm": True,  "use_attention": False, "n_heads": 8},
    "no_bilstm":     {"use_cnn": True,  "use_bilstm": False, "use_attention": True,  "n_heads": 8},
    "no_cnn":        {"use_cnn": False, "use_bilstm": True,  "use_attention": True,  "n_heads": 8},
    "no_cnn_no_attn":{"use_cnn": False, "use_bilstm": True,  "use_attention": False, "n_heads": 8},
    "single_head":   {"use_cnn": True,  "use_bilstm": True,  "use_attention": True,  "n_heads": 1, "head_dim": 512},
}


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def prepare_data(cfg):
    """One-time data prep so every ablation run trains on identical splits."""
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
    x_tr_flat, y_tr_flat = pre.fit_transform(train_df)
    x_te_flat, y_te_flat = pre.transform(test_df)

    from collections import Counter
    counts = Counter(y_tr_flat)
    k = max(1, min(5, min(counts.values()) - 1))
    smote = SMOTE(random_state=cfg.data.random_state, k_neighbors=k)
    x_tr_flat, y_tr_flat = smote.fit_resample(x_tr_flat, y_tr_flat)

    sb = SequenceBuilder(seq_len=cfg.data.sequence_length)
    x_train, y_train = sb.transform(x_tr_flat, y_tr_flat)
    x_test, y_test = sb.transform(x_te_flat, y_te_flat)

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train,
        test_size=cfg.data.val_ratio / (cfg.data.train_ratio + cfg.data.val_ratio),
        random_state=cfg.data.random_state, stratify=y_train,
    )
    return (x_tr, y_tr), (x_val, y_val), (x_test, y_test), x_train.shape[-1]


def run_one(cfg, name: str, model_kwargs: dict, data):
    (x_tr, y_tr), (x_val, y_val), (x_test, y_test), n_features = data

    train_loader = make_loader(x_tr, y_tr, cfg.train.batch_size, True, cfg.train.num_workers)
    val_loader = make_loader(x_val, y_val, cfg.train.batch_size, False, cfg.train.num_workers)
    test_loader = make_loader(x_test, y_test, cfg.train.batch_size, False, cfg.train.num_workers)

    set_seed(cfg.seed)  # reset before each variant for fairness
    model = CNNBiLSTMMHA(
        input_features=n_features, seq_len=cfg.data.sequence_length,
        cnn_channels=cfg.model.cnn_channels,
        cnn_kernel_size=cfg.model.cnn_kernel_size,
        cnn_pool_size=cfg.model.cnn_pool_size,
        lstm_hidden=cfg.model.lstm_hidden,
        lstm_layers=cfg.model.lstm_layers,
        lstm_dropout=cfg.model.lstm_dropout,
        head_dim=model_kwargs.get("head_dim", cfg.model.head_dim),
        fc_dims=cfg.model.fc_dims,
        fc_dropout=cfg.model.fc_dropout,
        num_classes=cfg.model.num_classes,
        n_heads=model_kwargs["n_heads"],
        use_cnn=model_kwargs["use_cnn"],
        use_bilstm=model_kwargs["use_bilstm"],
        use_attention=model_kwargs["use_attention"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Ablation: {name}  (params={n_params:,}) ===")

    trainer = Trainer(model, cfg.train, ckpt_dir=CHECKPOINT_DIR, run_name=f"abl_{name}")
    history = trainer.fit(train_loader, val_loader)

    y_true, y_pred = trainer.predict(test_loader)
    # Probabilities for AUC
    trainer.model.eval()
    probas = []
    with torch.no_grad():
        for xb, _ in test_loader:
            p = torch.softmax(trainer.model(xb.to(trainer.device)), dim=-1).cpu().numpy()
            probas.append(p)
    y_proba = np.concatenate(probas, axis=0)
    metrics = compute_metrics(y_true, y_pred, y_proba)

    sample = torch.randn(cfg.train.batch_size, cfg.data.sequence_length, n_features)
    lat_mean, lat_std = latency_benchmark(trainer.model, sample)

    return {
        "variant": name,
        "metrics": metrics.as_dict(),
        "latency_ms_mean": lat_mean,
        "latency_ms_std": lat_std,
        "n_params": int(n_params),
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    cfg = default_config()
    cfg.train.epochs = args.epochs
    set_seed(cfg.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Preparing TON_IoT data once for all {len(ABLATION_CONFIGS)} variants...")
    data = prepare_data(cfg)

    all_results = []
    for name, kw in ABLATION_CONFIGS.items():
        all_results.append(run_one(cfg, name, kw, data))

    # Δ vs full
    full = next(r for r in all_results if r["variant"] == "full")
    base_acc = full["metrics"]["accuracy"]
    for r in all_results:
        r["delta_acc_vs_full"] = r["metrics"]["accuracy"] - base_acc

    out = RESULTS_DIR / "ablation_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved -> {out}")

    print("\n=== Ablation summary ===")
    print(f"{'variant':18s} {'acc':>8s} {'f1':>8s} {'ΔAcc':>8s} {'params':>12s}")
    for r in all_results:
        print(f"{r['variant']:18s} "
              f"{r['metrics']['accuracy']*100:7.2f}% "
              f"{r['metrics']['f1_macro']*100:7.2f}% "
              f"{r['delta_acc_vs_full']*100:+7.2f}% "
              f"{r['n_params']:>12,}")


if __name__ == "__main__":
    main()
