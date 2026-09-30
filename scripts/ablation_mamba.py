"""
Ablation study for MambaNIDS (Q1 component-contribution analysis).

Variants
--------
    full       : Mamba + MHA + FDAttn      (proposed architecture)
    no_fdattn  : Mamba + MHA               (remove frequency branch)
    no_mha     : Mamba       + FDAttn      (remove time-attention)
    no_mamba   :              MHA + FDAttn (replace Mamba with identity, n_layers=0)

Each variant is trained on TON_IoT with the SAME data split, SAME seed,
SAME optimiser, SAME epochs — only the architecture differs. Results are
written to results/ablation_mamba.json.
"""
from __future__ import annotations

import io
import json
import random
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows so labels like "no FDAttn" never hit cp1256
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import default_config, CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR     # noqa: E402
from src.data import load_ton_iot, Preprocessor, SequenceBuilder              # noqa: E402
from src.data.loader_ton_iot import CLASS_LABELS as TON_LABELS                # noqa: E402
from src.evaluation import compute_metrics, latency_benchmark                 # noqa: E402
from src.models import MambaNIDS                                              # noqa: E402
from src.training import Trainer                                              # noqa: E402
from src.training.trainer import make_loader                                  # noqa: E402


VARIANTS = [
    # (key, label, kwargs)
    ("full",      "Full (Mamba+MHA+FDAttn)", dict(n_mamba_layers=2, use_mha=True,  use_freq_attention=True )),
    ("no_fdattn", "no FDAttn",               dict(n_mamba_layers=2, use_mha=True,  use_freq_attention=False)),
    ("no_mha",    "no MHA",                  dict(n_mamba_layers=2, use_mha=False, use_freq_attention=True )),
    ("no_mamba",  "no Mamba",                dict(n_mamba_layers=0, use_mha=True,  use_freq_attention=True )),
]
EPOCHS = 15


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def prepare_data(cfg):
    print("[data] loading TON_IoT")
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
    n_features = x_train.shape[-1]
    print(f"      n_features={n_features}, train={len(x_tr):,}, val={len(x_val):,}, test={len(x_test):,}")
    return (x_tr, y_tr), (x_val, y_val), (x_test, y_test), n_features


def train_variant(key: str, label: str, kwargs: dict, splits, n_features: int, cfg):
    print(f"\n=== variant: {key} | {label} ===")
    (x_tr, y_tr), (x_val, y_val), (x_test, y_test) = splits

    train_loader = make_loader(x_tr, y_tr, cfg.train.batch_size, True, cfg.train.num_workers)
    val_loader   = make_loader(x_val, y_val, cfg.train.batch_size, False, cfg.train.num_workers)
    test_loader  = make_loader(x_test, y_test, cfg.train.batch_size, False, cfg.train.num_workers)

    set_seed(cfg.seed)
    model = MambaNIDS(
        input_features=n_features, seq_len=cfg.data.sequence_length,
        d_model=256, num_classes=cfg.model.num_classes, **kwargs,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      trainable params: {n_params:,}")

    cfg.train.epochs = EPOCHS
    trainer = Trainer(model, cfg.train, ckpt_dir=CHECKPOINT_DIR, run_name=f"abl_{key}")
    t0 = time.time()
    history = trainer.fit(train_loader, val_loader)
    train_time = time.time() - t0

    # test eval
    y_true, y_pred = trainer.predict(test_loader)
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
        "variant": key,
        "label": label,
        "kwargs": {k: v for k, v in kwargs.items()},
        "metrics": metrics.as_dict(),
        "latency_ms_mean": lat_mean,
        "latency_ms_std": lat_std,
        "n_params": int(n_params),
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "train_time_s": train_time,
    }


def main():
    cfg = default_config()
    set_seed(cfg.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = RESULTS_DIR / "ablation_mamba.json"
    results = []

    # Resume support: load any previously completed variants
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
            done = {r["variant"] for r in results}
            print(f"[resume] already done: {sorted(done)}")
        except Exception:
            results = []
    done = {r["variant"] for r in results}

    splits_full = prepare_data(cfg)
    splits = splits_full[:3]
    n_features = splits_full[3]

    for key, label, kwargs in VARIANTS:
        if key in done:
            print(f"[skip] {key} already in {out_path.name}")
            continue
        rec = train_variant(key, label, kwargs, splits, n_features, cfg)
        results.append(rec)
        # incremental save (so a crash mid-run isn't fatal)
        out_path.write_text(json.dumps(results, indent=2))

    # compute deltas relative to full
    full_acc = next(r["metrics"]["accuracy"] for r in results if r["variant"] == "full")
    full_f1  = next(r["metrics"]["f1_macro"] for r in results if r["variant"] == "full")
    full_params = next(r["n_params"] for r in results if r["variant"] == "full")
    for r in results:
        r["delta_acc_vs_full"]    = r["metrics"]["accuracy"] - full_acc
        r["delta_f1_vs_full"]     = r["metrics"]["f1_macro"] - full_f1
        r["param_ratio_vs_full"]  = r["n_params"] / max(full_params, 1)

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
