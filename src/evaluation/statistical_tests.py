"""
Publication-grade statistical machinery for comparing classifiers.

Implements:
  * Wilcoxon signed-rank test (paired, non-parametric)
  * Paired bootstrap test on per-sample correctness
  * Bonferroni-corrected McNemar tests across a family of comparisons
  * Friedman test (omnibus, rank-based) + Nemenyi post-hoc + critical-difference
  * Cohen's d and odds-ratio effect sizes
  * Holm-Bonferroni step-down procedure

References
----------
Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets.
    Journal of Machine Learning Research, 7, 1-30.
Holm, S. (1979). A simple sequentially rejective multiple test procedure.
Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised
    classification learning algorithms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


# ----------------------------------------------------------------------- #
# Pairwise tests                                                          #
# ----------------------------------------------------------------------- #
def wilcoxon_paired(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test on paired per-sample loss (or accuracy) scores.

    Returns the statistic and a two-sided p-value. Uses the exact distribution
    if n < 50 (SciPy default).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores must be paired and same length")
    try:
        stat, p = stats.wilcoxon(scores_a, scores_b, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        # All differences are zero -> identical models
        return {"statistic": 0.0, "p_value": 1.0}
    return {"statistic": float(stat), "p_value": float(p)}


def paired_bootstrap_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired bootstrap test on per-sample correctness vectors (1 = correct).

    Returns the observed accuracy difference, the bootstrap mean, the 95% CI on
    the difference, and the one-sided p-value for H0: acc_a <= acc_b.
    """
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    if len(correct_b) != n:
        raise ValueError("correctness vectors must be same length")
    observed = float(correct_a.mean() - correct_b.mean())
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # One-sided p-value: probability under bootstrap that diff <= 0
    p_one = float((diffs <= 0).mean())
    return {
        "observed_diff": observed,
        "bootstrap_mean_diff": float(diffs.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value_one_sided": p_one,
    }


# ----------------------------------------------------------------------- #
# Multi-comparison correction                                             #
# ----------------------------------------------------------------------- #
def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[Dict]:
    """
    Holm-Bonferroni step-down procedure.

    Returns a list of dicts (one per input p-value) with the adjusted p-value
    and a boolean ``reject`` flag.
    """
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = [None] * n
    prev = 0.0
    for rank, idx in enumerate(order):
        # adjusted = min(1, (n - rank) * p_idx, previous_adjusted)
        adj = min(1.0, (n - rank) * p_values[idx])
        adj = max(adj, prev)
        adjusted[idx] = adj
        prev = adj
    return [
        {"p_raw": p_values[i], "p_adj": adjusted[i], "reject": adjusted[i] < alpha}
        for i in range(n)
    ]


def bonferroni(p_values: List[float], alpha: float = 0.05) -> List[Dict]:
    n = len(p_values)
    return [
        {"p_raw": p, "p_adj": min(1.0, p * n), "reject": min(1.0, p * n) < alpha}
        for p in p_values
    ]


# ----------------------------------------------------------------------- #
# Friedman + Nemenyi (multi-classifier, multi-dataset)                    #
# ----------------------------------------------------------------------- #
def friedman_test(scores: np.ndarray) -> Dict[str, float]:
    """
    Non-parametric omnibus test for K classifiers over N datasets.

    Parameters
    ----------
    scores : (N, K) array — rows = datasets, columns = classifiers (higher = better).
    """
    if scores.ndim != 2:
        raise ValueError("expected (N_datasets, K_classifiers) matrix")
    stat, p = stats.friedmanchisquare(*[scores[:, j] for j in range(scores.shape[1])])
    # Mean rank per classifier (lower rank = better when ranking by negative score)
    ranks = np.apply_along_axis(stats.rankdata, axis=1, arr=-scores)  # lower rank = better
    mean_ranks = ranks.mean(axis=0)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "mean_ranks": mean_ranks.tolist(),
    }


def nemenyi_cd(n_classifiers: int, n_datasets: int, alpha: float = 0.05) -> float:
    """
    Critical Difference for the Nemenyi post-hoc test.

    Uses the studentized range table approximation
    from Demšar (2006). For typical comparisons we hard-code q for α = 0.05.
    For very large K this is a conservative approximation.
    """
    # Studentized-range critical values q_{alpha, K, inf} for α = 0.05
    Q05 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    Q10 = {
        2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589,
        7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920,
    }
    table = Q05 if abs(alpha - 0.05) < 1e-6 else Q10
    q = table.get(n_classifiers, table[max(table)])
    cd = q * np.sqrt(n_classifiers * (n_classifiers + 1) / (6.0 * n_datasets))
    return float(cd)


# ----------------------------------------------------------------------- #
# Effect sizes                                                            #
# ----------------------------------------------------------------------- #
def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict[str, float]:
    """Standardised mean difference (paired, Hedges-corrected when small n)."""
    diff = scores_a - scores_b
    d = diff.mean() / (diff.std(ddof=1) + 1e-12)
    # Hedges' g correction for small samples
    n = len(diff)
    correction = 1.0 - 3.0 / (4.0 * n - 9.0) if n > 3 else 1.0
    g = d * correction
    return {"cohen_d": float(d), "hedges_g": float(g), "n": int(n)}


def interpret_d(d: float) -> str:
    """Cohen's qualitative bands."""
    ad = abs(d)
    if ad < 0.2:  return "negligible"
    if ad < 0.5:  return "small"
    if ad < 0.8:  return "medium"
    return "large"


def mcnemar_odds_ratio(b: int, c: int) -> Dict[str, float]:
    """
    Effect size for paired classification disagreement:
    odds_ratio = b / c, with Haldane–Anscombe ½-correction when either cell is 0.
    """
    b_, c_ = b + 0.5, c + 0.5
    or_ = b_ / c_
    log_or = np.log(or_)
    # Standard error of log OR (Wald approximation)
    se = np.sqrt(1.0 / b_ + 1.0 / c_)
    z = 1.96
    return {
        "odds_ratio": float(or_),
        "log_or": float(log_or),
        "ci_low": float(np.exp(log_or - z * se)),
        "ci_high": float(np.exp(log_or + z * se)),
    }
