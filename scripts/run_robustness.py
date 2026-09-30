"""
Run adversarial-robustness sweep on every trained checkpoint.

Outputs results/robustness.json with FGSM and PGD accuracy curves over a
shared epsilon grid for each model.  Adversarial examples are crafted in
the [0, 1] normalised feature space.
"""
from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import default_config, CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR    # noqa: E402
from src.data import load_ton_iot, Preprocessor, SequenceBuilder            # noqa: E402
from src.data.loader_ton_iot import CLASS_LABELS as TON_LABELS              # noqa: E402
from src.evaluation import fgsm_attack, pgd_attack, robustness_curve        # noqa: E402
from src.training.trainer import make_loader, _resolve_device               # noqa: E402
from scripts.train import build_model                                       # noqa: E402


MODELS = ["mamba_nids", "cnn_bilstm_mha", "transformer_ids", "cnn_lstm", "lstm", "gru"]
EPSILONS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]


def prepare_test_loader(cfg):
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
    loader = make_loader(x_test, y_test, 256, False, 0)
    return loader, x_test.shape[-1]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-batches", type=int, default=4,
                    help="evaluate on this many test batches to keep wall-clock manageable")
    ap.add_argument("--pgd-steps", type=int, default=10)
    args = ap.parse_args()

    cfg = default_config()
    device = _resolve_device(cfg.train.device)
    loader, n_features = prepare_test_loader(cfg)

    out_all = []
    for name in MODELS:
        ckpt = CHECKPOINT_DIR / f"{name}_best.pt"
        if not ckpt.exists():
            print(f"[skip] missing {ckpt}")
            continue
        print(f"\n=== robustness sweep: {name} ===")
        model = build_model(name, n_features, cfg.data.sequence_length, cfg.model)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))

        fgsm = robustness_curve(
            model, loader, fgsm_attack, EPSILONS, device,
            max_batches=args.max_batches, attack_name="FGSM",
        )
        pgd = robustness_curve(
            model, loader,
            partial(pgd_attack, n_steps=args.pgd_steps, random_start=True),
            EPSILONS, device,
            max_batches=args.max_batches, attack_name=f"PGD-{args.pgd_steps}",
        )
        out_all.append({
            "model": name,
            "clean_accuracy": fgsm.clean_accuracy,
            "n_samples": fgsm.n_samples,
            "fgsm": fgsm.as_dict(),
            "pgd": pgd.as_dict(),
        })
        print(f"  clean acc = {fgsm.clean_accuracy*100:.2f}%")
        for eps, acc_f, acc_p in zip(EPSILONS, fgsm.accuracies, pgd.accuracies):
            print(f"    eps={eps:.3f}  FGSM={acc_f*100:6.2f}%   PGD={acc_p*100:6.2f}%")

    out = {
        "models": out_all,
        "epsilons": EPSILONS,
        "pgd_steps": args.pgd_steps,
        "max_batches": args.max_batches,
    }
    path = RESULTS_DIR / "robustness.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
