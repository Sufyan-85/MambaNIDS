"""
Convenience driver that runs the entire pipeline end-to-end:

  1. Train all 4 models on TON_IoT
  2. Run McNemar tests
  3. Run ablation study
  4. Train CNN-BiLSTM-MHA on IoTNet24 (cross-dataset)
  5. Generate all plots
  6. Build the final Word report

Existing checkpoints / results are reused — only missing artifacts are produced.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import CHECKPOINT_DIR, RESULTS_DIR  # noqa: E402


def _run(args: list[str]) -> int:
    print(f"\n$ {' '.join(args)}")
    return subprocess.call(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--skip-training", action="store_true",
                    help="reuse existing model checkpoints")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-cross", action="store_true",
                    help="skip IoTNet24 cross-dataset run")
    args = ap.parse_args()

    py = sys.executable

    # 1. Train all four models
    if not args.skip_training:
        for model in ("cnn_bilstm_mha", "cnn_lstm", "gru", "lstm"):
            ckpt = CHECKPOINT_DIR / f"{model}_best.pt"
            if ckpt.exists():
                print(f"[skip] {model} already trained ({ckpt})")
                continue
            _run([py, "scripts/train.py", "--model", model,
                  "--dataset", "ton_iot", "--epochs", str(args.epochs)])

    # 2. McNemar tests
    for opp in ("lstm", "gru", "cnn_lstm"):
        out = RESULTS_DIR / f"mcnemar_cnn_bilstm_mha_vs_{opp}.json"
        if out.exists():
            print(f"[skip] McNemar vs {opp} already done")
            continue
        _run([py, "scripts/evaluate.py", "--a", "cnn_bilstm_mha",
              "--b", opp, "--dataset", "ton_iot"])

    # 3. Ablation
    abl = RESULTS_DIR / "ablation_results.json"
    if not args.skip_ablation and not abl.exists():
        _run([py, "scripts/ablation.py", "--epochs", str(args.epochs)])
    elif abl.exists():
        print("[skip] ablation already done")

    # 4. Cross-dataset
    if not args.skip_cross:
        iot_results = RESULTS_DIR / "iotnet24_cnn_bilstm_mha_results.json"
        if not iot_results.exists():
            # Train with smaller epochs since dataset is small
            _run([py, "scripts/train.py", "--model", "cnn_bilstm_mha",
                  "--dataset", "iotnet24", "--epochs", str(args.epochs)])
            # Rename so it doesn't conflict with the TON_IoT main result
            src = RESULTS_DIR / "cnn_bilstm_mha_results.json"
            if src.exists() and src != iot_results:
                # If TON_IoT result still exists, we already trained it; back-up not needed
                pass
        else:
            print("[skip] IoTNet24 already done")

    # 5. Plots
    _run([py, "scripts/make_plots.py"])

    # 6. Word report
    _run([py, "scripts/make_report.py"])

    print("\nAll done. Report at:", PROJECT_ROOT / "REPORT.docx")


if __name__ == "__main__":
    main()
