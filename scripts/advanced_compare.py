"""
Academic-grade comparison of every trained checkpoint.

Outputs a single JSON file ``results/advanced_comparison.json`` containing,
for each model:

  * Bootstrap 95% CIs on Accuracy / F1 / Cohen's κ / MCC
  * ECE and Brier
  * Per-class P/R/F1/support
  * Complexity (params / GFLOPs / throughput / GPU peak memory)
  * Cached predictions + probabilities for downstream plotting

It then computes:

  * Wilcoxon signed-rank tests between CNN-BiLSTM-MHA and each baseline
    on per-sample loss
  * McNemar χ² + odds ratio for every pair
  * Holm-Bonferroni correction across the family of comparisons
  * Friedman omnibus + Nemenyi CD across (TON_IoT, IoTNet24) where available

Run-time: re-uses existing checkpoints, only forward-pass cost (~30 s).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import default_config, CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR  # noqa: E402
from src.data import load_ton_iot, Preprocessor, SequenceBuilder           # noqa: E402
from src.data.loader_ton_iot import CLASS_LABELS as TON_LABELS              # noqa: E402
from src.evaluation import (                                                 # noqa: E402
    compute_metrics, compute_advanced, measure_complexity,
    mcnemar_test, mcnemar_odds_ratio, paired_bootstrap_test,
    wilcoxon_paired, holm_bonferroni, friedman_test, nemenyi_cd,
    cohens_d, interpret_d,
)
from src.training.trainer import make_loader, _resolve_device              # noqa: E402
from scripts.train import build_model                                       # noqa: E402


MODELS = ["cnn_bilstm_mha", "cnn_lstm", "gru", "lstm"]


def _per_sample_logloss(y_true: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
    """NLL per sample — used as the paired Wilcoxon score."""
    eps = 1e-12
    n = len(y_true)
    return -np.log(np.clip(y_proba[np.arange(n), y_true], eps, 1.0))


def _gather_predictions(cfg):
    """One pass per model → (model_name → (y_true, y_pred, y_proba))."""
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
    n_features = x_test.shape[-1]

    test_loader = make_loader(x_test, y_test, cfg.train.batch_size, False, 0)
    device = _resolve_device(cfg.train.device)

    preds_by_model = {}
    for name in MODELS:
        ckpt = CHECKPOINT_DIR / f"{name}_best.pt"
        if not ckpt.exists():
            print(f"[skip] missing {ckpt}")
            continue
        model = build_model(name, n_features, cfg.data.sequence_length, cfg.model)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model = model.to(device).eval()

        probas = []
        with torch.no_grad():
            for xb, _ in test_loader:
                p = torch.softmax(model(xb.to(device)), dim=-1).cpu().numpy()
                probas.append(p)
        y_proba = np.concatenate(probas, axis=0)
        y_pred = y_proba.argmax(axis=1)
        preds_by_model[name] = {
            "y_true": y_test, "y_pred": y_pred, "y_proba": y_proba, "model_obj": model,
        }
        del model
    return preds_by_model, n_features


def main():
    cfg = default_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] gathering predictions from every checkpoint...")
    preds, n_features = _gather_predictions(cfg)
    if not preds:
        print("No checkpoints found. Train models first.")
        return 1

    # --------- per-model advanced metrics ---------------------------- #
    print("[2/4] computing per-model advanced metrics + complexity probes...")
    per_model = {}
    for name, d in preds.items():
        adv = compute_advanced(
            d["y_true"], d["y_pred"], d["y_proba"],
            class_names=TON_LABELS, n_boot=1000, seed=42,
        )
        basic = compute_metrics(d["y_true"], d["y_pred"], d["y_proba"]).as_dict()
        cx = measure_complexity(
            d["model_obj"],
            input_shape=(cfg.train.batch_size, cfg.data.sequence_length, n_features),
            n_runs=100,
        )
        per_model[name] = {
            "basic": basic,
            "advanced": adv.as_dict(),
            "complexity": cx,
        }
        print(f"  {name:18s} acc={basic['accuracy']*100:.2f}% "
              f"k={adv.kappa:.3f}  MCC={adv.mcc:.3f}  "
              f"ECE={adv.ece:.3f}  Brier={adv.brier:.3f}  "
              f"GFLOPs={cx['gflops_per_forward']:.2f}")

    # --------- pairwise statistics vs the proposed model ------------- #
    print("[3/4] pairwise tests (proposed vs each baseline)...")
    proposed = "cnn_bilstm_mha"
    if proposed not in preds:
        print(f"proposed model {proposed!r} not trained — aborting pairwise stats")
        out = RESULTS_DIR / "advanced_comparison.json"
        out.write_text(json.dumps({"per_model": per_model}, indent=2, default=float))
        return 0

    proposed_d = preds[proposed]
    ll_a = _per_sample_logloss(proposed_d["y_true"], proposed_d["y_proba"])
    pairwise = []
    raw_p_values_mcnemar = []
    raw_p_values_wilcox = []
    for name, d in preds.items():
        if name == proposed:
            continue
        # McNemar with odds ratio
        mc = mcnemar_test(d["y_true"], proposed_d["y_pred"], d["y_pred"])
        odds = mcnemar_odds_ratio(mc["b"], mc["c"])
        # Wilcoxon on paired NLL
        ll_b = _per_sample_logloss(d["y_true"], d["y_proba"])
        wx = wilcoxon_paired(ll_a, ll_b)
        # Paired bootstrap on correctness
        ca = (proposed_d["y_pred"] == proposed_d["y_true"]).astype(float)
        cb = (d["y_pred"] == d["y_true"]).astype(float)
        bs = paired_bootstrap_test(ca, cb, n_boot=2000, seed=7)
        # Effect size on per-sample NLL difference
        eff = cohens_d(ll_b, ll_a)  # positive d -> baseline has higher loss
        eff["interpretation"] = interpret_d(eff["cohen_d"])

        pairwise.append({
            "baseline": name,
            "mcnemar": {**mc, "odds_ratio": odds},
            "wilcoxon_paired_nll": wx,
            "paired_bootstrap_correctness": bs,
            "effect_size_nll": eff,
        })
        raw_p_values_mcnemar.append(mc["p_value"])
        raw_p_values_wilcox.append(wx["p_value"])

    # Multiple-comparison corrections (Holm-Bonferroni)
    holm_mc = holm_bonferroni(raw_p_values_mcnemar)
    holm_wx = holm_bonferroni(raw_p_values_wilcox)
    for i, entry in enumerate(pairwise):
        entry["mcnemar_holm"] = holm_mc[i]
        entry["wilcoxon_holm"] = holm_wx[i]

    # --------- Friedman/Nemenyi across available datasets ----------- #
    print("[4/4] Friedman + Nemenyi across datasets...")
    # We collect per-model accuracy on each available dataset's results.json
    ds_accs = {}  # dataset_name -> {model: acc}
    for jpath in RESULTS_DIR.glob("*_results.json"):
        if jpath.name.startswith("mcnemar"): continue
        if jpath.name in {"ablation_results.json", "advanced_comparison.json"}: continue
        if jpath.name == "cnn_bilstm_mha_iotnet24_results.json":
            ds_accs.setdefault("iotnet24", {})["cnn_bilstm_mha"] = json.loads(jpath.read_text())["metrics"]["accuracy"]
        elif jpath.name == "cnn_bilstm_mha_ton_iot_results.json":
            ds_accs.setdefault("ton_iot", {})["cnn_bilstm_mha"] = json.loads(jpath.read_text())["metrics"]["accuracy"]
        else:
            # Generic _results.json -> TON_IoT model run
            model_name = jpath.stem.replace("_results", "")
            if model_name in MODELS:
                ds_accs.setdefault("ton_iot", {})[model_name] = json.loads(jpath.read_text())["metrics"]["accuracy"]

    # Build (datasets x models) matrix
    common_models = [m for m in MODELS if all(m in ds_accs[d] for d in ds_accs)]
    friedman = None
    if len(ds_accs) >= 2 and len(common_models) >= 2:
        rows = []
        for d in ds_accs:
            rows.append([ds_accs[d][m] for m in common_models])
        matrix = np.array(rows)
        try:
            fr = friedman_test(matrix)
            cd = nemenyi_cd(len(common_models), matrix.shape[0])
            friedman = {
                "datasets": list(ds_accs.keys()),
                "models": common_models,
                "score_matrix": matrix.tolist(),
                **fr,
                "critical_difference": cd,
            }
            print(f"  Friedman χ² = {fr['statistic']:.2f}  p = {fr['p_value']:.4g}  CD = {cd:.3f}")
        except ValueError as e:
            print(f"  Friedman skipped: {e}")

    out = {
        "per_model": per_model,
        "pairwise_vs_proposed": pairwise,
        "friedman_nemenyi": friedman,
        "class_names": TON_LABELS,
    }
    path = RESULTS_DIR / "advanced_comparison.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
