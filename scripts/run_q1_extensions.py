"""
Sequence-runner: executes the remaining Q1 strengthening experiments
after adversarial training of MambaNIDS has completed.

Steps
-----
1. Robustness sweep on the adversarially-trained model.
2. Train all six baselines on N-BaIoT (cross-dataset table).
3. Train all baselines on IoTNet24 (binary task table).
4. Multi-seed training of the top four models on TON_IoT
   (3 seeds, 15 epochs each).

Skipped automatically if the resulting artefact already exists.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR, CHECKPOINT_DIR  # noqa: E402

PY = sys.executable

BASELINES = ["mamba_nids", "cnn_bilstm_mha", "transformer_ids",
             "cnn_lstm", "gru", "lstm"]
TOP_MODELS = ["mamba_nids", "cnn_bilstm_mha", "transformer_ids", "cnn_lstm"]
SEEDS = [42, 7, 123]


def _run(args, desc):
    print(f"\n==> {desc}\n$ {' '.join(args)}")
    t0 = time.time()
    p = subprocess.run(args, cwd=str(PROJECT_ROOT))
    print(f"  ({time.time()-t0:.0f}s, exit={p.returncode})")
    return p.returncode == 0


def _move_result(model: str, dataset: str) -> Path | None:
    """train.py overwrites results/<model>_results.json — move it to a unique name."""
    src = RESULTS_DIR / f"{model}_results.json"
    dst = RESULTS_DIR / f"{model}_{dataset}_results.json"
    if src.exists():
        dst.write_bytes(src.read_bytes())
        return dst
    return None


def step_adv_robustness():
    """Adversarial sweep on the adv-trained MambaNIDS, compared with clean models."""
    out = RESULTS_DIR / "robustness_adv.json"
    if out.exists():
        print(f"[skip] {out} already exists")
        return
    # Temporarily move clean mamba_nids checkpoint aside so the eval picks up adv
    clean_ckpt = CHECKPOINT_DIR / "mamba_nids_best.pt"
    adv_ckpt = CHECKPOINT_DIR / "mamba_nids_adv_best.pt"
    backup = CHECKPOINT_DIR / "mamba_nids_clean_backup.pt"
    if not adv_ckpt.exists():
        print(f"[skip] no adv checkpoint at {adv_ckpt}")
        return
    if clean_ckpt.exists():
        clean_ckpt.rename(backup)
    adv_ckpt.read_bytes()
    (CHECKPOINT_DIR / "mamba_nids_best.pt").write_bytes(adv_ckpt.read_bytes())

    try:
        _run([PY, "scripts/run_robustness.py", "--max-batches", "6", "--pgd-steps", "10"],
             "robustness sweep on adv-trained MambaNIDS")
        # Move output
        src = RESULTS_DIR / "robustness.json"
        if src.exists():
            out.write_bytes(src.read_bytes())
    finally:
        # Restore clean checkpoint
        if backup.exists():
            (CHECKPOINT_DIR / "mamba_nids_best.pt").write_bytes(backup.read_bytes())
            backup.unlink()


def step_cross_dataset(dataset: str, epochs: int = 15):
    """Train every baseline on the given dataset for cross-dataset comparison."""
    for model in BASELINES:
        out_existing = RESULTS_DIR / f"{model}_{dataset}_results.json"
        if out_existing.exists():
            print(f"[skip] {model}/{dataset} already done")
            continue
        ok = _run(
            [PY, "scripts/train.py", "--model", model,
             "--dataset", dataset, "--epochs", str(epochs)],
            f"train {model} on {dataset}",
        )
        if ok:
            _move_result(model, dataset)


def step_multiseed(epochs: int = 15):
    """Run scripts/multiseed.py for our top models with 3 seeds."""
    _run(
        [PY, "scripts/multiseed.py",
         "--models", *TOP_MODELS,
         "--datasets", "ton_iot",
         "--seeds", *map(str, SEEDS),
         "--epochs", str(epochs)],
        "multi-seed training",
    )


def main():
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    step_adv_robustness()
    step_cross_dataset("nbaiot", epochs=15)
    step_cross_dataset("iotnet24", epochs=15)
    step_multiseed(epochs=15)
    print("\nAll Q1 extensions complete.")


if __name__ == "__main__":
    main()
