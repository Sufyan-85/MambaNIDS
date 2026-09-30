"""
Empirical PAC-Bayes generalisation bound for a trained classifier.

We use the McAllester (1999) bound:

    R(Q) <= R_emp(Q) + sqrt( ( KL(Q || P) + ln(2 sqrt(m) / delta) ) / (2m) )

where
    Q       = posterior over weights (Gaussian centred at theta_hat, isotropic var sigma_post^2)
    P       = prior (Gaussian centred at 0, isotropic var sigma_prior^2)
    m       = training-sample count
    delta   = failure probability (we use 0.05 for a 95% bound)
    R(Q)    = expected 0-1 loss of the stochastic classifier
    R_emp(Q)= empirical 0-1 loss of the stochastic classifier on the train set

KL divergence between two isotropic Gaussians of equal dimension D collapses to

    KL(N(theta,sigma_q^2 I) || N(0,sigma_p^2 I))
      = (D/2) * [ log(sigma_p^2/sigma_q^2) - 1 + sigma_q^2/sigma_p^2 ]
        + ||theta||^2 / (2 sigma_p^2)

References
----------
McAllester, D. A. (1999). PAC-Bayesian model averaging.
Dziugaite & Roy (2017). Computing nonvacuous generalization bounds for
    deep stochastic neural networks with many more parameters than training data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PacBayesResult:
    empirical_risk_stochastic: float
    kl_divergence: float
    bound: float          # the McAllester upper bound on R(Q)
    n_samples: int
    n_params: int
    sigma_prior: float
    sigma_posterior: float
    delta: float


def kl_isotropic_gaussian(
    theta: torch.Tensor, sigma_q: float, sigma_p: float
) -> float:
    """KL between N(theta, sigma_q^2 I) and N(0, sigma_p^2 I)."""
    D = theta.numel()
    log_ratio = 2.0 * (math.log(sigma_p) - math.log(sigma_q))
    trace_term = (sigma_q ** 2) / (sigma_p ** 2)
    mean_term = (theta.pow(2).sum() / (sigma_p ** 2)).item()
    return 0.5 * (D * (log_ratio - 1.0 + trace_term) + mean_term)


@torch.no_grad()
def _stochastic_empirical_risk(
    model: nn.Module,
    train_loader,
    sigma_posterior: float,
    n_mc_samples: int = 5,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> float:
    """
    Monte-Carlo estimate of 0-1 loss of the stochastic classifier
    Q = N(theta_hat, sigma_post^2 I).
    """
    device = device or next(model.parameters()).device
    model = model.to(device).eval()
    # Snapshot weights so we can perturb in-place and restore
    original = {k: p.detach().clone() for k, p in model.named_parameters()}

    losses = []
    rng = torch.Generator(device=device).manual_seed(0)

    n_batches_done = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        batch_errors = []
        for _ in range(n_mc_samples):
            for k, p in model.named_parameters():
                noise = torch.randn(p.shape, generator=rng, device=device)
                p.data = original[k] + sigma_posterior * noise
            preds = model(xb).argmax(dim=-1)
            batch_errors.append((preds != yb).float().mean().item())
        losses.append(float(np.mean(batch_errors)))
        n_batches_done += 1
        if max_batches is not None and n_batches_done >= max_batches:
            break

    # restore weights
    for k, p in model.named_parameters():
        p.data = original[k]
    return float(np.mean(losses)) if losses else float("nan")


def mcallester_bound(
    empirical_risk: float, kl: float, m: int, delta: float = 0.05
) -> float:
    """sqrt(( KL + ln(2 sqrt(m)/delta) ) / (2 m)) added to empirical risk."""
    log_term = math.log(2.0 * math.sqrt(m) / delta)
    return empirical_risk + math.sqrt((kl + log_term) / (2 * m))


def pac_bayes_evaluation(
    model: nn.Module,
    train_loader,
    sigma_prior: float = 0.1,
    sigma_posterior: float = 0.01,
    n_mc_samples: int = 5,
    delta: float = 0.05,
    max_batches: Optional[int] = None,
) -> PacBayesResult:
    """
    One-shot McAllester PAC-Bayes bound estimation for a trained model.

    Parameters
    ----------
    sigma_prior : float
        Standard deviation of the prior N(0, sigma_prior^2 I). Smaller priors
        give tighter bounds when the trained weights are also small, but the
        prior MUST NOT depend on the training data.
    sigma_posterior : float
        Standard deviation of the diagonal Gaussian posterior centred at
        theta_hat. Smaller -> closer to the deterministic classifier but
        higher KL.
    """
    # Flatten all parameters into a single vector
    theta = torch.cat([p.detach().flatten() for p in model.parameters()])
    n_params = theta.numel()
    n_samples = sum(len(b[0]) for b in train_loader)

    kl = kl_isotropic_gaussian(theta.cpu(), sigma_posterior, sigma_prior)
    risk = _stochastic_empirical_risk(
        model, train_loader, sigma_posterior,
        n_mc_samples=n_mc_samples,
        max_batches=max_batches,
    )
    bound = mcallester_bound(risk, kl, n_samples, delta)

    return PacBayesResult(
        empirical_risk_stochastic=risk,
        kl_divergence=kl,
        bound=bound,
        n_samples=n_samples,
        n_params=n_params,
        sigma_prior=sigma_prior,
        sigma_posterior=sigma_posterior,
        delta=delta,
    )
