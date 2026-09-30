"""
Publication-grade metrics that complement the basic block in metrics.py:

  * Cohen's Kappa κ (chance-corrected agreement)
  * Matthews Correlation Coefficient (MCC) — robust to class imbalance
  * Expected Calibration Error (ECE) with 15 equal-width bins
  * Brier score (multiclass)
  * Bootstrap 95% confidence intervals for any metric
  * Per-class precision / recall / F1 / support table

These follow Demšar (2006), Bouckaert & Frank (2004), and Guo et al. (2017).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, brier_score_loss, cohen_kappa_score, f1_score,
    matthews_corrcoef, precision_recall_fscore_support,
)


# ----------------------------------------------------------------------- #
# Cohen's κ + MCC                                                         #
# ----------------------------------------------------------------------- #
def kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred))


def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(matthews_corrcoef(y_true, y_pred))


# ----------------------------------------------------------------------- #
# Expected Calibration Error (Guo et al., 2017)                            #
# ----------------------------------------------------------------------- #
def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Multi-class ECE.

    Predicted confidence = max of softmax row.
    Bin samples by confidence, return weighted mean of |accuracy − confidence|.
    """
    if y_proba.ndim == 1 or y_proba.shape[1] < 2:
        return float("nan")
    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences <= hi)
        if mask.any():
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


# ----------------------------------------------------------------------- #
# Multiclass Brier score                                                   #
# ----------------------------------------------------------------------- #
def multiclass_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Mean squared error between one-hot ground truth and predicted probas."""
    n, k = y_proba.shape
    onehot = np.zeros_like(y_proba)
    onehot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))


# ----------------------------------------------------------------------- #
# Bootstrap confidence intervals                                          #
# ----------------------------------------------------------------------- #
def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Returns (point_estimate, ci_lower, ci_upper) for a metric.

    Uses percentile bootstrap with `n_boot` resamples of the test set.
    `metric_fn(y_true, y_pred)` must return a float.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = float(metric_fn(y_true, y_pred))
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            samples[i] = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            samples[i] = np.nan
    samples = samples[~np.isnan(samples)]
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return point, lo, hi


# ----------------------------------------------------------------------- #
# Per-class precision / recall / F1 / support                              #
# ----------------------------------------------------------------------- #
def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> List[Dict[str, float]]:
    """Return a list of dicts, one per class, with P/R/F1/support."""
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0,
    )
    return [
        {
            "class": class_names[i],
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(supp[i]),
        }
        for i in range(len(class_names))
    ]


# ----------------------------------------------------------------------- #
# Bundle                                                                   #
# ----------------------------------------------------------------------- #
@dataclass
class AdvancedResult:
    kappa: float
    kappa_ci: Tuple[float, float]
    mcc: float
    mcc_ci: Tuple[float, float]
    accuracy_ci: Tuple[float, float]
    f1_macro_ci: Tuple[float, float]
    ece: float
    brier: float
    per_class: List[Dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "kappa": self.kappa,
            "kappa_ci_low": self.kappa_ci[0],
            "kappa_ci_high": self.kappa_ci[1],
            "mcc": self.mcc,
            "mcc_ci_low": self.mcc_ci[0],
            "mcc_ci_high": self.mcc_ci[1],
            "accuracy_ci_low": self.accuracy_ci[0],
            "accuracy_ci_high": self.accuracy_ci[1],
            "f1_macro_ci_low": self.f1_macro_ci[0],
            "f1_macro_ci_high": self.f1_macro_ci[1],
            "ece": self.ece,
            "brier": self.brier,
            "per_class": self.per_class,
        }


def compute_advanced(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    n_boot: int = 1000,
    seed: int = 42,
) -> AdvancedResult:
    """One-shot advanced-metric computation with bootstrap CIs."""

    def _acc(a, b): return accuracy_score(a, b)
    def _f1(a, b):  return f1_score(a, b, average="macro", zero_division=0)
    def _kappa(a, b): return cohen_kappa_score(a, b)
    def _mcc(a, b):   return matthews_corrcoef(a, b)

    _, lo_a, hi_a = bootstrap_ci(_acc, y_true, y_pred, n_boot, seed=seed)
    _, lo_f, hi_f = bootstrap_ci(_f1,  y_true, y_pred, n_boot, seed=seed + 1)
    p_k, lo_k, hi_k = bootstrap_ci(_kappa, y_true, y_pred, n_boot, seed=seed + 2)
    p_m, lo_m, hi_m = bootstrap_ci(_mcc,   y_true, y_pred, n_boot, seed=seed + 3)

    ece = expected_calibration_error(y_true, y_proba) if y_proba is not None else float("nan")
    brier = multiclass_brier(y_true, y_proba) if y_proba is not None else float("nan")
    per_cls = per_class_report(y_true, y_pred, class_names)

    return AdvancedResult(
        kappa=p_k, kappa_ci=(lo_k, hi_k),
        mcc=p_m,   mcc_ci=(lo_m, hi_m),
        accuracy_ci=(lo_a, hi_a),
        f1_macro_ci=(lo_f, hi_f),
        ece=ece, brier=brier,
        per_class=per_cls,
    )
