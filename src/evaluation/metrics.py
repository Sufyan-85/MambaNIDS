"""
Evaluation metrics matching the paper (§6.2):
  Accuracy, Precision (macro), Recall (macro), F1-score (macro),
  AUC-ROC, False Positive Rate, McNemar's test for pairwise significance,
  and inference-latency benchmarking.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)


@dataclass
class MetricResult:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    auc_roc_ovr: float
    fpr: float
    confusion: np.ndarray

    def as_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "auc_roc_ovr": self.auc_roc_ovr,
            "fpr": self.fpr,
        }


def _false_positive_rate(cm: np.ndarray) -> float:
    """
    Macro-averaged False Positive Rate.

    For each class c: FPR_c = FP_c / (FP_c + TN_c).
    """
    n_classes = cm.shape[0]
    rates = []
    total = cm.sum()
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        denom = fp + tn
        rates.append(fp / denom if denom > 0 else 0.0)
    return float(np.mean(rates))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
) -> MetricResult:
    """All-in-one metrics block matching Table 3 of the paper."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    fpr = _false_positive_rate(cm)

    if y_proba is not None and y_proba.ndim == 2:
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")
    else:
        auc = float("nan")

    return MetricResult(
        accuracy=float(acc),
        precision_macro=float(prec),
        recall_macro=float(rec),
        f1_macro=float(f1),
        auc_roc_ovr=float(auc),
        fpr=float(fpr),
        confusion=cm,
    )


@torch.no_grad()
def latency_benchmark(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    n_warmup: int = 20,
    n_runs: int = 200,
) -> Tuple[float, float]:
    """
    Return (mean_ms, std_ms) of inference latency on the given batch.

    Use a representative batch size (paper uses 512).
    """
    device = next(model.parameters()).device
    model.eval()
    sample_input = sample_input.to(device)

    # Warm-up
    for _ in range(n_warmup):
        model(sample_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(timings)), float(np.std(timings))


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    """
    McNemar's test (§6.2) for paired classifier comparison.

    Returns the contingency cells and the p-value (chi-square with 1 dof,
    continuity-corrected). A small p-value (< 0.05) means the two models
    have a statistically significant difference in errors.
    """
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    # b = A correct, B wrong; c = A wrong, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}

    # Continuity-corrected chi-square statistic
    stat = ((abs(b - c) - 1) ** 2) / n
    # 1-dof chi-square survival function (closed-form via erfc)
    from math import erfc, sqrt
    p_value = float(erfc(sqrt(stat / 2.0)))
    return {"b": b, "c": c, "statistic": float(stat), "p_value": p_value}
