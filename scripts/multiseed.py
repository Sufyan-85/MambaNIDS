"""
Multi-seed training runner — academic-grade variance estimation.

For each requested (model, dataset, seed) tuple:
  1. Set every relevant RNG to ``seed``.
  2. Train the model from scratch for ``--epochs`` epochs.
  3. Evaluate on the held-out test split with bootstrap CIs.
  4. Save per-seed metrics under ``results/multiseed/<dataset>/<model>/seed_<n>.json``.

After all seeds finish, aggregate into ``results/multiseed_summary.json`` with
mean ± std for every metric.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR  # noqa: E402


DEFAULT_MODELS = ["mamba_nids", "cnn_bilstm_mha", "transformer_ids",
                  "cnn_lstm", "gru", "lstm"]
DEFAULT_SEEDS = [42, 7, 123]
MULTI_DIR = RESULTS_DIR / "multiseed"


def _run_one(model: str, dataset: str, seed: int, epochs: int) -> dict | None:
    """Run scripts/train.py with the requested config; return parsed metrics."""
    out_dir = MULTI_DIR / dataset / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_{seed}.json"
    if out_path.exists():
        print(f"[skip] {model}/{dataset}/seed={seed} already done")
        return json.loads(out_path.read_text())

    print(f"\n--- {model} on {dataset}, seed {seed}, {epochs} epochs ---")
    cmd = [
        sys.executable, "scripts/train.py",
        "--model", model,
        "--dataset", dataset,
        "--epochs", str(epochs),
    ]
    # Train script reads cfg.seed from config.py — we pass seed via env override
    import os
    env = {**os.environ, "MULTISEED_SEED": str(seed)}
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          cwd=str(PROJECT_ROOT))
    duration = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL] {model}/{dataset}/seed={seed}: {proc.stderr[-500:]}")
        return None

    # train.py saves to results/<model>_results.json; move to per-seed file
    src = RESULTS_DIR / f"{model}_results.json"
    if not src.exists():
        print(f"[FAIL] no results file for {model}")
        return None
    result = json.loads(src.read_text())
    result["seed"] = seed
    result["duration_seconds"] = round(duration, 1)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  acc={result['metrics']['accuracy']*100:.2f}%  "
          f"f1={result['metrics']['f1_macro']*100:.2f}%  "
          f"time={duration:.0f}s")
    return result


def _aggregate() -> dict:
    """Walk results/multiseed/ and produce mean ± std per (model, dataset)."""
    summary = {}
    if not MULTI_DIR.exists():
        return summary
    for dataset_dir in MULTI_DIR.iterdir():
        if not dataset_dir.is_dir(): continue
        dataset = dataset_dir.name
        summary.setdefault(dataset, {})
        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir(): continue
            model = model_dir.name
            per_seed = []
            for jpath in sorted(model_dir.glob("seed_*.json")):
                per_seed.append(json.loads(jpath.read_text()))
            if not per_seed:
                continue
            seeds = [r["seed"] for r in per_seed]
            metrics_keys = list(per_seed[0]["metrics"].keys())
            agg = {}
            for k in metrics_keys:
                vals = np.array([r["metrics"][k] for r in per_seed], dtype=float)
                agg[k] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "values": [float(v) for v in vals],
                }
            lat_vals = np.array([r["latency_ms_mean"] for r in per_seed], dtype=float)
            agg["latency_ms"] = {
                "mean": float(lat_vals.mean()),
                "std": float(lat_vals.std(ddof=1)) if len(lat_vals) > 1 else 0.0,
            }
            summary[dataset][model] = {"seeds": seeds, "n_runs": len(per_seed), **agg}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--datasets", nargs="+", default=["ton_iot"])
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip training, just rebuild the summary file")
    args = ap.parse_args()

    MULTI_DIR.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        for ds in args.datasets:
            for model in args.models:
                for seed in args.seeds:
                    _run_one(model, ds, seed, args.epochs)

    summary = _aggregate()
    out = RESULTS_DIR / "multiseed_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out}")

    print("\n=== multi-seed summary ===")
    for ds, models in summary.items():
        print(f"\n[{ds}]")
        for m, r in models.items():
            acc = r.get("accuracy", {})
            f1 = r.get("f1_macro", {})
            print(f"  {m:18s}  n={r['n_runs']}  "
                  f"acc={acc.get('mean', 0)*100:6.2f} +/- {acc.get('std', 0)*100:5.2f}%  "
                  f"f1={f1.get('mean', 0)*100:6.2f} +/- {f1.get('std', 0)*100:5.2f}%")


if __name__ == "__main__":
    main()
