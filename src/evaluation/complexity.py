"""
Computational complexity instrumentation:

  * FLOPs per inference (via torch profiler hooks — module-level estimation)
  * Throughput (samples / second) at a representative batch size
  * Peak GPU memory consumption per inference call

These provide the Pareto-frontier data needed for an academic
accuracy-vs-cost analysis (Tan & Le, 2019; Howard et al., 2017).
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

import torch
import torch.nn as nn


# ----------------------------------------------------------------------- #
# FLOPs — manual layer-by-layer estimation                                #
# ----------------------------------------------------------------------- #
def _conv1d_flops(layer: nn.Conv1d, in_len: int) -> int:
    out_len = (in_len + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
    return 2 * layer.in_channels * layer.out_channels * layer.kernel_size[0] * out_len


def _linear_flops(layer: nn.Linear, seq_len: int = 1) -> int:
    return 2 * layer.in_features * layer.out_features * seq_len


def _lstm_flops(layer: nn.LSTM, seq_len: int) -> int:
    """4 gates × (input + hidden) inputs × hidden outputs × 2 ops × directions."""
    H = layer.hidden_size
    inp = layer.input_size
    dirs = 2 if layer.bidirectional else 1
    per_step = 4 * H * (inp + H + 1) * 2  # 2 for multiply + add
    base = per_step * seq_len * dirs * layer.num_layers
    return base


def _attn_flops(embed_dim: int, seq_len: int, n_heads: int) -> int:
    # QKV projections (3) + output projection (1) + attention QK^T + AV
    proj = 4 * 2 * embed_dim * embed_dim * seq_len   # linear projections
    attn = 2 * 2 * seq_len * seq_len * embed_dim     # QK^T then weighted sum
    return proj + attn


@torch.no_grad()
def estimate_flops(model: nn.Module, sample_input: torch.Tensor) -> int:
    """
    Layer-by-layer FLOP estimate for a single forward pass over a batch.

    Walks all named modules and adds the cost of every Conv1D, LSTM, GRU,
    Linear, and MultiheadAttention found. Other layers (BN, dropout, pool)
    contribute negligibly and are ignored.
    """
    B, T, _F = sample_input.shape
    total = 0
    cur_len = T

    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            total += B * _conv1d_flops(m, cur_len)
        elif isinstance(m, nn.MaxPool1d):
            cur_len = max(1, cur_len // m.kernel_size if isinstance(m.kernel_size, int)
                          else cur_len // m.kernel_size[0])
        elif isinstance(m, nn.LSTM):
            total += B * _lstm_flops(m, cur_len)
        elif isinstance(m, nn.GRU):
            # GRU = 3 gates instead of 4 → 3/4 of LSTM cost
            H = m.hidden_size
            inp = m.input_size
            dirs = 2 if m.bidirectional else 1
            per_step = 3 * H * (inp + H + 1) * 2
            total += B * per_step * cur_len * dirs * m.num_layers
        elif isinstance(m, nn.MultiheadAttention):
            total += B * _attn_flops(m.embed_dim, cur_len, m.num_heads)
        elif isinstance(m, nn.Linear):
            total += B * _linear_flops(m, 1)

    return int(total)


# ----------------------------------------------------------------------- #
# Throughput + peak memory                                                #
# ----------------------------------------------------------------------- #
@torch.no_grad()
def throughput_and_memory(
    model: nn.Module,
    sample_input: torch.Tensor,
    n_runs: int = 200,
    n_warmup: int = 20,
) -> Dict[str, float]:
    """
    Returns:
      throughput_samples_per_sec
      peak_memory_mb (CUDA only, otherwise nan)
      latency_ms_mean
    """
    device = next(model.parameters()).device
    model.eval()
    x = sample_input.to(device)
    B = x.shape[0]

    # Warm up
    for _ in range(n_warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    # Time
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_samples = B * n_runs
    throughput = total_samples / elapsed
    latency_ms = (elapsed / n_runs) * 1000.0

    peak_mem_mb = float("nan")
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "throughput_samples_per_sec": float(throughput),
        "latency_ms_mean": float(latency_ms),
        "peak_memory_mb": float(peak_mem_mb),
        "batch_size": int(B),
        "device": str(device),
    }


# ----------------------------------------------------------------------- #
# Bundle                                                                   #
# ----------------------------------------------------------------------- #
def measure_complexity(
    model: nn.Module,
    input_shape: Tuple[int, int, int],
    n_runs: int = 200,
) -> Dict[str, float]:
    """One-call complexity probe — FLOPs + throughput + memory + params."""
    sample = torch.randn(*input_shape)
    flops = estimate_flops(model, sample)
    sample = sample.to(next(model.parameters()).device)
    perf = throughput_and_memory(model, sample, n_runs=n_runs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "n_params": int(n_params),
        "flops_per_forward": int(flops),
        "gflops_per_forward": float(flops / 1e9),
        **perf,
    }
