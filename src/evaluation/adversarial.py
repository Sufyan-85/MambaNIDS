"""
Adversarial-robustness evaluation suite.

Implements:
  * Fast Gradient Sign Method (Goodfellow et al., 2015)
  * Projected Gradient Descent (Madry et al., 2018)
  * Carlini-Wagner L2 attack (Carlini & Wagner, 2017) — simplified version
  * Adversarial training defence wrapper
  * Per-epsilon robustness curve helper

All attacks operate on the same (B, T, F) tensor format used by the other
models. Inputs are assumed to lie in [0, 1] (min-max normalised by the
preprocessor).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Per-attack implementations                                                  #
# --------------------------------------------------------------------------- #
def _attack_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass tolerant of cuDNN LSTM/GRU backward restrictions.

    cuDNN's fused RNN implementation cannot backward in eval mode, so we
    temporarily disable cudnn for the attack forward pass. Dropout layers
    are also forced into eval mode to keep the attack deterministic.
    """
    was_training = model.training
    # Force eval semantics for dropout / batchnorm
    model.eval()
    with torch.backends.cudnn.flags(enabled=False):
        out = model(x)
    if was_training:
        model.train()
    return out


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    clamp: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """
    Fast Gradient Sign Method.

    Returns adversarial input x_adv s.t. ||x_adv - x||_inf <= epsilon.
    """
    x_adv = x.clone().detach().requires_grad_(True)
    logits = _attack_forward(model, x_adv)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    perturb = epsilon * x_adv.grad.sign()
    x_adv = (x + perturb).clamp(*clamp).detach()
    return x_adv


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    step_size: float = None,
    n_steps: int = 20,
    clamp: Tuple[float, float] = (0.0, 1.0),
    random_start: bool = True,
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD-L∞).

    Default step size = epsilon / 4 if not provided.
    """
    if step_size is None:
        step_size = epsilon / 4
    x_adv = x.clone().detach()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = x_adv.clamp(*clamp)

    for _ in range(n_steps):
        x_adv.requires_grad_(True)
        logits = _attack_forward(model, x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.sign()
        # Project back into the epsilon ball around x
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = x_adv.clamp(*clamp)

    return x_adv.detach()


def cw_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1.0,
    n_steps: int = 100,
    lr: float = 0.01,
    clamp: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """
    Carlini-Wagner L2 (simplified, untargeted).

    Solves min ||delta||_2^2 + c * f(x + delta)
    where f is the margin loss.
    """
    delta = torch.zeros_like(x, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    n_classes = _attack_forward(model, x).shape[-1]
    y_onehot = F.one_hot(y, n_classes).float()

    for _ in range(n_steps):
        x_adv = (x + delta).clamp(*clamp)
        logits = _attack_forward(model, x_adv)
        true_logit = (logits * y_onehot).sum(dim=-1)
        other_max = (logits - 1e9 * y_onehot).max(dim=-1).values
        # f(x') = max(true - other_max, 0) — push other class above true
        f_loss = torch.clamp(true_logit - other_max, min=0.0)
        loss = (delta.pow(2).sum(dim=tuple(range(1, x.ndim))) + c * f_loss).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (x + delta).clamp(*clamp).detach()


# --------------------------------------------------------------------------- #
# Adversarial-training defence                                                #
# --------------------------------------------------------------------------- #
def adversarial_training_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epsilon: float = 0.05,
    pgd_steps: int = 7,
    pgd_step_size: float = 0.01,
) -> float:
    """
    One Madry-style adversarial-training step.

    Generates PGD examples then trains on the union of clean+adversarial
    cross-entropy loss.
    """
    model.eval()
    x_adv = pgd_attack(model, x, y, epsilon=epsilon, step_size=pgd_step_size,
                       n_steps=pgd_steps)
    model.train()
    optimizer.zero_grad()
    logits_clean = model(x)
    logits_adv = model(x_adv)
    loss = 0.5 * (F.cross_entropy(logits_clean, y) + F.cross_entropy(logits_adv, y))
    loss.backward()
    optimizer.step()
    return loss.item()


# --------------------------------------------------------------------------- #
# Per-epsilon curve evaluation                                                #
# --------------------------------------------------------------------------- #
@dataclass
class RobustnessResult:
    attack: str
    epsilons: List[float]
    accuracies: List[float]
    clean_accuracy: float
    n_samples: int

    def as_dict(self) -> Dict:
        return {
            "attack": self.attack,
            "epsilons": self.epsilons,
            "accuracies": self.accuracies,
            "clean_accuracy": self.clean_accuracy,
            "n_samples": self.n_samples,
        }


@torch.no_grad()
def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return (model(x).argmax(dim=-1) == y).float().mean().item()


def robustness_curve(
    model: nn.Module,
    loader,
    attack_fn: Callable,
    epsilons: List[float],
    device: torch.device,
    attack_kwargs: Optional[Dict] = None,
    max_batches: Optional[int] = None,
    attack_name: str = "attack",
) -> RobustnessResult:
    """
    Sweep ``attack_fn`` over ``epsilons`` and report accuracy at each level.

    ``attack_fn(model, x, y, epsilon, **attack_kwargs) -> x_adv``.
    """
    attack_kwargs = attack_kwargs or {}
    # Collect a fixed evaluation batch by exhausting (or sub-sampling) the loader
    all_x, all_y = [], []
    for i, (xb, yb) in enumerate(loader):
        all_x.append(xb)
        all_y.append(yb)
        if max_batches is not None and i + 1 >= max_batches:
            break
    x_all = torch.cat(all_x).to(device)
    y_all = torch.cat(all_y).to(device)

    model = model.to(device)
    clean_acc = _accuracy(model, x_all, y_all)
    accs = []
    for eps in epsilons:
        x_adv = attack_fn(model, x_all, y_all, eps, **attack_kwargs)
        accs.append(_accuracy(model, x_adv, y_all))
    return RobustnessResult(
        attack=attack_name,
        epsilons=list(epsilons),
        accuracies=accs,
        clean_accuracy=clean_acc,
        n_samples=len(x_all),
    )


# --------------------------------------------------------------------------- #
# Certified robustness via randomised smoothing (Cohen et al., 2019)          #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def randomised_smoothing_predict(
    model: nn.Module,
    x: torch.Tensor,
    sigma: float = 0.05,
    n_samples: int = 100,
) -> torch.Tensor:
    """
    Smoothed classifier: predict the majority class under Gaussian noise N(0, sigma I).

    Returns the predicted class per sample (B,).
    """
    model.eval()
    B = x.shape[0]
    # Sample n_samples noisy versions and majority-vote
    votes = torch.zeros(B, dtype=torch.long, device=x.device)
    n_classes = None
    counts = None
    for i in range(n_samples):
        noise = torch.randn_like(x) * sigma
        logits = model((x + noise).clamp(0.0, 1.0))
        if counts is None:
            n_classes = logits.shape[-1]
            counts = torch.zeros(B, n_classes, device=x.device)
        preds = logits.argmax(dim=-1)
        counts.scatter_add_(1, preds.unsqueeze(-1), torch.ones_like(preds, dtype=torch.float).unsqueeze(-1))
    return counts.argmax(dim=-1)
