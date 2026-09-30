"""
Compare two trained models with McNemar's test on the held-out test split.

Usage
-----
    python scripts/evaluate.py --a cnn_bilstm_mha --b lstm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import default_config, CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR  # noqa: E402
from src.data import load_nsl_kdd, load_ton_iot, Preprocessor, SequenceBuilder  # noqa: E402
from src.data.loader import CLASS_LABELS as NSL_LABELS                      # noqa: E402
from src.data.loader_ton_iot import CLASS_LABELS as TON_LABELS              # noqa: E402
from src.evaluation import mcnemar_test                                    # noqa: E402
from src.training.trainer import make_loader, _resolve_device              # noqa: E402
from scripts.train import build_model                                      # noqa: E402


def load_model(name: str, n_features: int, seq_len: int, cfg) -> torch.nn.Module:
    model = build_model(name, n_features, seq_len, cfg.model)
    ckpt = CHECKPOINT_DIR / f"{name}_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"missing checkpoint {ckpt}. Train it first with `python scripts/train.py --model {name}`"
        )
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    return model


def get_predictions(model: torch.nn.Module, loader, device: torch.device) -> np.ndarray:
    model = model.to(device).eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="first model name")
    parser.add_argument("--b", required=True, help="second model name")
    parser.add_argument("--dataset", choices=("nsl_kdd", "ton_iot"), default="ton_iot")
    args = parser.parse_args()

    cfg = default_config()
    device = _resolve_device(cfg.train.device)

    # Rebuild test set deterministically (same seed/splits as train.py)
    if args.dataset == "nsl_kdd":
        train_df, test_df = load_nsl_kdd(DATA_DIR)
        categorical_cols = None
        class_labels = NSL_LABELS
    else:
        ton_csv = DATA_DIR / "TON_IoT datasets" / "Train_Test_Network_dataset" / "train_test_network.csv"
        train_df, test_df, categorical_cols = load_ton_iot(
            ton_csv, test_ratio=cfg.data.test_ratio, random_state=cfg.data.random_state,
        )
        class_labels = TON_LABELS
    cfg.model.num_classes = len(class_labels)

    pre = Preprocessor(
        mifs_threshold=cfg.data.mifs_threshold,
        random_state=cfg.data.random_state,
        categorical_cols=categorical_cols,
    )
    pre.fit_transform(train_df)
    x_test_flat, y_test_flat = pre.transform(test_df)
    sb = SequenceBuilder(seq_len=cfg.data.sequence_length)
    x_test, y_test = sb.transform(x_test_flat, y_test_flat)

    test_loader = make_loader(x_test, y_test, cfg.train.batch_size, False, 0)

    model_a = load_model(args.a, x_test.shape[-1], cfg.data.sequence_length, cfg)
    model_b = load_model(args.b, x_test.shape[-1], cfg.data.sequence_length, cfg)
    pred_a = get_predictions(model_a, test_loader, device)
    pred_b = get_predictions(model_b, test_loader, device)

    result = mcnemar_test(y_test, pred_a, pred_b)
    result["model_a"] = args.a
    result["model_b"] = args.b
    result["acc_a"] = float((pred_a == y_test).mean())
    result["acc_b"] = float((pred_b == y_test).mean())

    print(json.dumps(result, indent=2))
    out = RESULTS_DIR / f"mcnemar_{args.a}_vs_{args.b}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
