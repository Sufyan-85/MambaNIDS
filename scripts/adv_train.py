"""
Madry-style adversarial training for any model in our registry.

For each batch we (i) craft a PGD-N adversary, (ii) compute the
half-clean + half-adversarial cross-entropy, and (iii) update on the
sum. The result is stored as ``<model>_adv_best.pt`` so it does not
overwrite the clean checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import default_config, CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR    # noqa: E402
from src.data import load_ton_iot, Preprocessor, SequenceBuilder            # noqa: E402
from src.data.loader_ton_iot import CLASS_LABELS as TON_LABELS              # noqa: E402
from src.evaluation import compute_metrics, latency_benchmark, pgd_attack   # noqa: E402
from src.evaluation.adversarial import _attack_forward                       # noqa: E402
from src.training.trainer import make_loader, _resolve_device               # noqa: E402
from scripts.train import build_model                                       # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def prepare(cfg):
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
    return (x_tr, y_tr), (x_val, y_val), (x_test, y_test), x_train.shape[-1]


def train_epoch(model, loader, optim, device, epsilon, pgd_steps, pgd_step_size,
                clean_weight=0.5):
    """Half-clean half-adversarial cross-entropy."""
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        # Craft adversary in eval mode (clean gradients), then train on both
        x_adv = pgd_attack(model, xb, yb, epsilon=epsilon,
                           step_size=pgd_step_size, n_steps=pgd_steps,
                           random_start=True)
        model.train()
        optim.zero_grad()
        logits_clean = model(xb)
        logits_adv   = model(x_adv)
        loss = clean_weight * F.cross_entropy(logits_clean, yb) + \
               (1 - clean_weight) * F.cross_entropy(logits_adv, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_clean(model, loader, device):
    model.eval()
    correct, n = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.backends.cudnn.flags(enabled=False):
            correct += (model(xb).argmax(dim=-1) == yb).sum().item()
        n += xb.size(0)
    return correct / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mamba_nids")
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--pgd-steps", type=int, default=7)
    ap.add_argument("--pgd-step-size", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--clean-weight", type=float, default=0.5)
    args = ap.parse_args()

    cfg = default_config()
    set_seed(cfg.seed)
    device = _resolve_device(cfg.train.device)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] preparing TON_IoT")
    (x_tr, y_tr), (x_val, y_val), (x_test, y_test), n_features = prepare(cfg)
    train_loader = make_loader(x_tr, y_tr, cfg.train.batch_size, True, cfg.train.num_workers)
    val_loader   = make_loader(x_val, y_val, cfg.train.batch_size, False, cfg.train.num_workers)
    test_loader  = make_loader(x_test, y_test, cfg.train.batch_size, False, cfg.train.num_workers)

    print(f"[2/4] building {args.model}")
    model = build_model(args.model, n_features, cfg.data.sequence_length, cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      params: {n_params:,}")

    optim = Adam(model.parameters(), lr=cfg.train.learning_rate,
                 betas=(cfg.train.beta1, cfg.train.beta2),
                 weight_decay=cfg.train.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=args.epochs)

    best_val, best_state = 0.0, None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, train_loader, optim, device,
                           args.epsilon, args.pgd_steps, args.pgd_step_size,
                           args.clean_weight)
        sched.step()
        clean = eval_clean(model, val_loader, device)
        print(f"  ep {ep:3d}/{args.epochs} | loss={loss:.4f}  val_clean={clean*100:.2f}%  ({time.time()-t0:.0f}s)")
        if clean > best_val:
            best_val = clean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"[3/4] best val_clean={best_val*100:.2f}%; saving adv-trained checkpoint")
    ckpt = CHECKPOINT_DIR / f"{args.model}_adv_best.pt"
    torch.save(best_state, ckpt)
    print(f"      saved -> {ckpt}")

    # Restore best for test
    model.load_state_dict(best_state)
    model = model.to(device).eval()

    print(f"[4/4] clean test evaluation")
    preds, probas = [], []
    for xb, yb in test_loader:
        xb = xb.to(device)
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            p = torch.softmax(model(xb), dim=-1).cpu().numpy()
        probas.append(p); preds.append(p.argmax(axis=1))
    y_pred = np.concatenate(preds); y_proba = np.concatenate(probas)
    metrics = compute_metrics(y_test, y_pred, y_proba)
    sample = torch.randn(cfg.train.batch_size, cfg.data.sequence_length, n_features)
    lat_mean, lat_std = latency_benchmark(model, sample)

    summary = {
        "model": args.model,
        "training_mode": "adversarial",
        "epsilon": args.epsilon,
        "pgd_steps": args.pgd_steps,
        "epochs": args.epochs,
        "clean_weight": args.clean_weight,
        "metrics": metrics.as_dict(),
        "latency_ms_mean": lat_mean,
        "latency_ms_std": lat_std,
        "n_params": int(n_params),
    }
    out = RESULTS_DIR / f"{args.model}_adv_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
