"""
Pure-PyTorch implementation of the Mamba (S6) selective state-space block.

Reference
---------
Gu, A., & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective
State Spaces. arXiv:2312.00752.

Notation (sequence length L, model dim D, state dim N, expansion E):
    x : (B, L, D) input
    z : (B, L, E*D) gate
    -> SiLU(z) * SSM(Conv1d(x_proj))
    -> output_projection -> (B, L, D)

This implementation uses a sequential cumulative scan rather than the
hardware-aware parallel scan from the original CUDA kernel. It is slower
than `mamba-ssm` but cross-platform and dependency-free.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# RMSNorm                                                                     #
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# --------------------------------------------------------------------------- #
# Selective Scan (sequential reference implementation)                        #
# --------------------------------------------------------------------------- #
def selective_scan(
    u: torch.Tensor,      # (B, L, D_in)
    delta: torch.Tensor,  # (B, L, D_in)  — input-dependent step size
    A: torch.Tensor,      # (D_in, N)     — log-parameterised; we use -exp(A_log)
    B: torch.Tensor,      # (B, L, N)     — input-dependent
    C: torch.Tensor,      # (B, L, N)     — input-dependent
    D: torch.Tensor,      # (D_in,)       — skip connection
) -> torch.Tensor:
    """
    Implements the recurrence
        h_t = exp(delta_t * A) * h_{t-1} + (delta_t * B_t) * u_t
        y_t = C_t @ h_t + D * u_t

    Returns y of shape (B, L, D_in).
    """
    Bsz, L, D_in = u.shape
    N = A.shape[-1]

    # Discretise A and B per (batch, time, channel, state).
    # Clamp the exponent to [-30, 0] to avoid both vanishing (exp(-large) -> 0)
    # and overflowing (exp(positive) -> inf). A is constructed to be negative.
    exponent = (delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)).clamp(min=-30.0, max=0.0)
    deltaA = torch.exp(exponent)
    # deltaB_u: (B, L, D_in, N) = delta[..., None] * B[..., None, :] * u[..., None]
    deltaB_u = (delta.unsqueeze(-1) * B.unsqueeze(2)) * u.unsqueeze(-1)

    # Cumulative scan (sequential — O(L) but cross-platform)
    h = torch.zeros(Bsz, D_in, N, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = deltaA[:, t] * h + deltaB_u[:, t]                # (B, D_in, N)
        y_t = torch.einsum("bdn,bn->bd", h, C[:, t])         # (B, D_in)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)                               # (B, L, D_in)

    # Add skip connection
    y = y + u * D.unsqueeze(0).unsqueeze(0)
    return y


# --------------------------------------------------------------------------- #
# Mamba Block                                                                 #
# --------------------------------------------------------------------------- #
class MambaBlock(nn.Module):
    """
    Selective-state-space block.

    Parameters
    ----------
    d_model : int
        Hidden dimension (input/output channel count).
    d_state : int
        SSM state-space dimension N.  Paper default: 16.
    d_conv : int
        Local Conv1D kernel size.  Paper default: 4.
    expand : int
        Inner-projection expansion factor.  Paper default: 2.
    bias : bool
        Use bias in projections.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        # delta requires its own low-rank projection of size dt_rank
        self.dt_rank = max(1, math.ceil(d_model / 16))

        # Input projection: x and z gates
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Causal 1D convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # SSM parameter projections (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Time-independent A and skip D
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        # We learn log(A) so A stays negative after exponentiation
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # Standard delta_proj bias initialisation (paper appendix)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D)."""
        B_, L, _ = x.shape

        # Project to inner dim and split into (x_inner, z gate)
        xz = self.in_proj(x)                              # (B, L, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)                  # each (B, L, d_inner)

        # Causal Conv1D — keeps the first L timesteps
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :L]
        x_conv = F.silu(x_conv.transpose(1, 2))           # (B, L, d_inner)

        # SSM parameter projections (input-dependent)
        x_dbl = self.x_proj(x_conv)                       # (B, L, dt_rank + 2N)
        delta, B_state, C_state = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # Cap delta to avoid huge step sizes when the model is poorly conditioned.
        delta = F.softplus(self.dt_proj(delta)).clamp(max=10.0)   # (B, L, d_inner)

        # Constrain A_log so A stays in a stable range
        A = -torch.exp(self.A_log.float().clamp(min=-10.0, max=10.0))    # (d_inner, N)
        y = selective_scan(
            x_conv, delta, A, B_state, C_state, self.D.float()
        )                                                  # (B, L, d_inner)

        # Gate and project
        y = y * F.silu(z)
        return self.out_proj(y)


# --------------------------------------------------------------------------- #
# Residual stack                                                              #
# --------------------------------------------------------------------------- #
class MambaResidualLayer(nn.Module):
    """Pre-RMSNorm + Mamba + residual."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class MambaStack(nn.Module):
    """N stacked Mamba residual layers + final RMSNorm."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaResidualLayer(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
