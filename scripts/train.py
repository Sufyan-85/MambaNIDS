"""
End-to-end training script.

Usage
-----
    python scripts/train.py --model cnn_bilstm_mha
    python scripts/train.py --model lstm
    python scripts/train.py --model gru
    python scripts/train.py --model cnn_lstm
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Make `import config` and `import src.*` work when this script is run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import default_config, CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR  # noqa: E402
from src.data import (                                                      # noqa: E402
    load_nsl_kdd, load_ton_iot, load_iotnet24, load_nbaiot,
    Preprocessor, SequenceBuilder,
)
from src.data.loader_ton_iot import CLASS_LABELS as TON_LABELS              # noqa: E402
from src.data.loader_iotnet24 import CLASS_LABELS as IOT_LABELS             # noqa: E402
from src.data.loader_nbaiot import CLASS_LABELS as NBAIOT_LABELS            # noqa: E402
from src.data.loader import CLASS_LABELS as NSL_LABELS                      # noqa: E402
from src.models import (                                                    # noqa: E402
    CNNBiLSTMMHA, VanillaLSTM, VanillaGRU, CNNLSTM,
    MambaNIDS, TransformerIDS, DeepAutoEncoder,
)
from src.training import Trainer                                           # noqa: E402
from src.training.trainer import make_loader                               # noqa: E402
from src.evaluation import compute_metrics, latency_benchmark              # noqa: E402


MODEL_REGISTRY = {
    "mamba_nids": "MambaNIDS (proposed Q1)",
    "cnn_bilstm_mha": "CNN-BiLSTM-MHA",
    "transformer_ids": "Transformer-IDS",
    "cnn_lstm": "CNN-LSTM",
    "lstm": "Vanilla LSTM",
    "gru": "Vanilla GRU",
    "autoencoder": "Deep AutoEncoder",
}

DATASETS = ("nsl_kdd", "ton_iot", "iotnet24", "nbaiot")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name: str, input_features: int, seq_len: int, model_cfg):
    if name == "mamba_nids":
        return MambaNIDS(
            input_features=input_features, seq_len=seq_len,
            d_model=256, n_mamba_layers=2,
            num_classes=model_cfg.num_classes,
        )
    if name == "cnn_bilstm_mha":
        return CNNBiLSTMMHA(
            input_features=input_features, seq_len=seq_len,
            cnn_channels=model_cfg.cnn_channels,
            cnn_kernel_size=model_cfg.cnn_kernel_size,
            cnn_pool_size=model_cfg.cnn_pool_size,
            lstm_hidden=model_cfg.lstm_hidden,
            lstm_layers=model_cfg.lstm_layers,
            lstm_dropout=model_cfg.lstm_dropout,
            n_heads=model_cfg.n_heads,
            head_dim=model_cfg.head_dim,
            fc_dims=model_cfg.fc_dims,
            fc_dropout=model_cfg.fc_dropout,
            num_classes=model_cfg.num_classes,
        )
    if name == "transformer_ids":
        return TransformerIDS(
            input_features=input_features, seq_len=seq_len,
            num_classes=model_cfg.num_classes,
        )
    if name == "autoencoder":
        return DeepAutoEncoder(
            input_features=input_features, seq_len=seq_len,
            num_classes=model_cfg.num_classes,
        )
    if name == "lstm":
        return VanillaLSTM(input_features=input_features, num_classes=model_cfg.num_classes)
    if name == "gru":
        return VanillaGRU(input_features=input_features, num_classes=model_cfg.num_classes)
    if name == "cnn_lstm":
        return CNNLSTM(
            input_features=input_features, seq_len=seq_len,
            num_classes=model_cfg.num_classes,
        )
    raise ValueError(f"unknown model: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="cnn_bilstm_mha")
    parser.add_argument("--dataset", choices=DATASETS, default="ton_iot")
    parser.add_argument("--epochs", type=int, default=None, help="override config.epochs")
    parser.add_argument("--limit-rows", type=int, default=None,
                        help="subsample to this many rows total (debug runs)")
    parser.add_argument("--no-smote", action="store_true", help="disable SMOTE oversampling")
    args = parser.parse_args()

    cfg = default_config()
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    # Allow scripts/multiseed.py to override via env without CLI plumbing
    import os
    env_seed = os.environ.get("MULTISEED_SEED")
    if env_seed is not None:
        cfg.seed = int(env_seed)
        cfg.data.random_state = int(env_seed)

    set_seed(cfg.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. load & preprocess ---------------------------------------- #
    print(f"[1/5] loading {args.dataset}")
    if args.dataset == "nsl_kdd":
        train_df, test_df = load_nsl_kdd(DATA_DIR)
        categorical_cols = None  # use NSL-KDD default
        class_labels = NSL_LABELS
    elif args.dataset == "ton_iot":
        ton_csv = DATA_DIR / "TON_IoT datasets" / "Train_Test_Network_dataset" / "train_test_network.csv"
        train_df, test_df, categorical_cols = load_ton_iot(
            ton_csv,
            test_ratio=cfg.data.test_ratio,
            random_state=cfg.data.random_state,
        )
        class_labels = TON_LABELS
    elif args.dataset == "iotnet24":
        iot_csv = DATA_DIR / "IoTNet24 Dataset" / "data.csv"
        train_df, test_df, categorical_cols = load_iotnet24(
            iot_csv,
            test_ratio=cfg.data.test_ratio,
            random_state=cfg.data.random_state,
        )
        class_labels = IOT_LABELS
    else:  # nbaiot
        nb_dir = DATA_DIR / "N-BaIoT"
        train_df, test_df, categorical_cols = load_nbaiot(
            nb_dir,
            rows_per_file=2000,
            test_ratio=cfg.data.test_ratio,
            random_state=cfg.data.random_state,
        )
        class_labels = NBAIOT_LABELS

    # Optional row cap for quick experiments
    if args.limit_rows:
        # stratified subsample so we keep small classes around
        from sklearn.model_selection import train_test_split as _tts
        train_keep = min(int(args.limit_rows * 0.85), len(train_df))
        test_keep = min(int(args.limit_rows * 0.15), len(test_df))
        train_df, _ = _tts(train_df, train_size=train_keep,
                           stratify=train_df["y"], random_state=cfg.data.random_state)
        test_df, _ = _tts(test_df, train_size=test_keep,
                          stratify=test_df["y"], random_state=cfg.data.random_state)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
    print(f"      train rows={len(train_df):,} | test rows={len(test_df):,}")
    cfg.model.num_classes = len(class_labels)

    print("[2/5] preprocessing (impute -> clip -> minmax -> MIFS)")
    pre = Preprocessor(
        mifs_threshold=cfg.data.mifs_threshold,
        random_state=cfg.data.random_state,
        categorical_cols=categorical_cols,
    )
    x_train_flat, y_train_flat = pre.fit_transform(train_df)
    x_test_flat, y_test_flat = pre.transform(test_df)
    n_features = x_train_flat.shape[1]
    print(f"      selected features: {n_features}")

    # ---- 2. SMOTE on flat training data ------------------------------ #
    use_smote = cfg.data.use_smote and not args.no_smote
    if use_smote:
        print("[3/5] SMOTE oversampling (training only)")
        # SMOTE needs at least 6 samples per class by default
        from collections import Counter
        counts = Counter(y_train_flat)
        min_count = min(counts.values())
        k = max(1, min(5, min_count - 1))
        smote = SMOTE(random_state=cfg.data.random_state, k_neighbors=k)
        x_train_flat, y_train_flat = smote.fit_resample(x_train_flat, y_train_flat)
        print(f"      after SMOTE: {Counter(y_train_flat)}")
    else:
        print("[3/5] SMOTE skipped")

    # ---- 3. build sliding windows ------------------------------------ #
    print("[4/5] building sliding windows (T=%d)" % cfg.data.sequence_length)
    sb = SequenceBuilder(seq_len=cfg.data.sequence_length)
    x_train, y_train = sb.transform(x_train_flat, y_train_flat)
    x_test, y_test = sb.transform(x_test_flat, y_test_flat)

    # split off a validation set from the training windows
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train,
        test_size=cfg.data.val_ratio / (cfg.data.train_ratio + cfg.data.val_ratio),
        random_state=cfg.data.random_state,
        stratify=y_train,
    )
    print(f"      train={len(x_tr):,} val={len(x_val):,} test={len(x_test):,}")

    train_loader = make_loader(x_tr, y_tr, cfg.train.batch_size, True, cfg.train.num_workers)
    val_loader = make_loader(x_val, y_val, cfg.train.batch_size, False, cfg.train.num_workers)
    test_loader = make_loader(x_test, y_test, cfg.train.batch_size, False, cfg.train.num_workers)

    # ---- 4. build model & train -------------------------------------- #
    print(f"[5/5] training {MODEL_REGISTRY[args.model]}")
    model = build_model(args.model, n_features, cfg.data.sequence_length, cfg.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      trainable params: {n_params:,}")

    trainer = Trainer(model, cfg.train, ckpt_dir=CHECKPOINT_DIR, run_name=args.model)
    history = trainer.fit(train_loader, val_loader)

    # ---- 5. evaluate on test ----------------------------------------- #
    print("\n=== Test evaluation ===")
    y_true, y_pred = trainer.predict(test_loader)
    # Get probabilities for AUC
    trainer.model.eval()
    probas = []
    with torch.no_grad():
        for xb, _ in test_loader:
            p = torch.softmax(trainer.model(xb.to(trainer.device)), dim=-1).cpu().numpy()
            probas.append(p)
    y_proba = np.concatenate(probas, axis=0)
    metrics = compute_metrics(y_true, y_pred, y_proba)

    # Latency on a paper-sized batch
    sample_batch = torch.randn(cfg.train.batch_size, cfg.data.sequence_length, n_features)
    lat_mean, lat_std = latency_benchmark(trainer.model, sample_batch)

    summary = {
        "model": args.model,
        "metrics": metrics.as_dict(),
        "latency_ms_mean": lat_mean,
        "latency_ms_std": lat_std,
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "n_features": int(n_features),
        "n_params": int(n_params),
    }
    print(json.dumps(summary, indent=2))

    out = RESULTS_DIR / f"{args.model}_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
