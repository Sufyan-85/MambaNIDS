"""
MambaNIDS — proposed architecture for the Q1 submission.

Pipeline
--------
    x (B, T, F)
      -> 1D-CNN stack (local feature extraction)
      -> Mamba stack  (selective state-space temporal modelling, O(L))
      -> Multi-Head Self-Attention + Frequency-Domain Attention (parallel)
      -> Global Average Pool
      -> Classification head

Novelty over CNN-BiLSTM-MHA
---------------------------
1. BiLSTM (O(L), 4 gates) is replaced by Mamba (O(L), single selective scan)
   with comparable expressivity but ~2x fewer parameters and faster inference.
2. A novel Frequency-Domain Attention branch runs in parallel with the
   standard MHA. It computes attention over the FFT of the temporal hidden
   state, capturing periodic flow signatures that purely time-domain
   attention misses.
3. A learnable gating coefficient (sigmoid) fuses the time-domain and
   frequency-domain attention outputs adaptively per token.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import MambaStack


# --------------------------------------------------------------------------- #
# Frequency-Domain Attention                                                  #
# --------------------------------------------------------------------------- #
class FrequencyDomainAttention(nn.Module):
    """
    Computes attention on the magnitude spectrum of the temporal hidden state.

    Steps
    -----
    1. FFT along the time axis -> complex tensor (B, T, D//2+1) per channel
       group. We use the real FFT (rfft) for efficiency.
    2. Take the magnitude as the spectral key/value.
    3. Single-head self-attention in the frequency domain.
    4. Inverse map back to time domain via learnable linear projection.
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)."""
        B, T, D = x.shape

        # rFFT along time. Magnitude only (phase info dropped — robust to
        # phase noise common in resampled flow features).
        freq = torch.fft.rfft(x, dim=1)            # (B, T//2+1, D), complex
        mag = freq.abs()                           # (B, T//2+1, D)
        Tf = mag.shape[1]

        # Multi-head attention over Tf frequency bins
        q = self.q_proj(mag).view(B, Tf, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(mag).view(B, Tf, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(mag).view(B, Tf, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out_f = attn @ v                           # (B, h, Tf, head_dim)
        out_f = out_f.transpose(1, 2).contiguous().view(B, Tf, D)

        # Pad / interpolate back to T frequency bins, then iFFT
        # Reattach the original phase so the inverse transform is real.
        phase = torch.angle(freq)
        out_complex = torch.polar(out_f, phase)
        out_t = torch.fft.irfft(out_complex, n=T, dim=1)
        return self.out_proj(out_t)


# --------------------------------------------------------------------------- #
# MambaNIDS                                                                   #
# --------------------------------------------------------------------------- #
class MambaNIDS(nn.Module):
    """
    Proposed Q1 architecture.

    Parameters
    ----------
    input_features : int
        F — flow features per time step.
    seq_len : int
        T — sequence length (typically 10).
    cnn_channels : list[int]
        Filter counts per Conv1D block. Default [64, 128, 256].
    d_model : int
        Hidden dimension fed into Mamba and the attention branches.
    n_mamba_layers : int
        Stacked Mamba residual blocks. Default 2.
    d_state, d_conv, expand : Mamba hyperparameters (see mamba.py).
    n_heads : int
        Multi-head attention heads in the time-domain branch.
    fc_dims : list[int]
        Hidden sizes in the classification head.
    num_classes : int
        Output classes.
    use_freq_attention : bool
        Disable to ablate the novel frequency branch.
    use_mha : bool
        Disable to ablate the time-domain attention.
    """

    def __init__(
        self,
        input_features: int,
        seq_len: int,
        cnn_channels: List[int] = (64, 128, 256),
        cnn_kernel_size: int = 3,
        cnn_pool_size: int = 2,
        d_model: int = 256,
        n_mamba_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 8,
        fc_dims: List[int] = (256, 128),
        fc_dropout: float = 0.4,
        num_classes: int = 10,
        use_freq_attention: bool = True,
        use_mha: bool = True,
    ):
        super().__init__()
        self.input_features = input_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_freq_attention = use_freq_attention
        self.use_mha = use_mha

        # ----- 1D-CNN stack -----
        blocks = []
        prev = input_features
        for out_c in cnn_channels:
            blocks += [
                nn.Conv1d(prev, out_c, kernel_size=cnn_kernel_size,
                          padding=cnn_kernel_size // 2),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(cnn_pool_size) if cnn_pool_size > 1 else nn.Identity(),
            ]
            prev = out_c
        self.cnn = nn.Sequential(*blocks)
        self.cnn_out = cnn_channels[-1]

        # ----- Project to d_model if needed -----
        self.cnn_to_model = (
            nn.Linear(self.cnn_out, d_model) if self.cnn_out != d_model
            else nn.Identity()
        )

        # ----- Mamba temporal stack -----
        self.mamba = MambaStack(
            d_model=d_model, n_layers=n_mamba_layers,
            d_state=d_state, d_conv=d_conv, expand=expand,
        )

        # ----- Attention branches -----
        if use_mha:
            self.mha = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=n_heads, batch_first=True,
            )
            self.mha_norm = nn.LayerNorm(d_model)
        else:
            self.mha = None
            self.mha_norm = nn.Identity()

        if use_freq_attention:
            self.freq_attn = FrequencyDomainAttention(d_model, n_heads=n_heads // 2)
            self.freq_norm = nn.LayerNorm(d_model)
            # Learnable per-token gate between time- and frequency-domain
            self.fusion_gate = nn.Linear(d_model, 1)
        else:
            self.freq_attn = None
            self.freq_norm = nn.Identity()
            self.fusion_gate = None

        # ----- Classification head -----
        head, prev = [], d_model
        for h in fc_dims:
            head += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(fc_dropout)]
            prev = h
        head.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*head)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) -> logits (B, num_classes)."""
        # 1D-CNN expects (B, C, T)
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)   # (B, T', C_last)
        h = self.cnn_to_model(h)                          # (B, T', d_model)

        # Mamba temporal stack
        h = self.mamba(h)                                 # (B, T', d_model)

        # Parallel attention branches (residual + LayerNorm in each)
        a_time = torch.zeros_like(h)
        a_freq = torch.zeros_like(h)
        if self.mha is not None:
            attn_out, _ = self.mha(h, h, h, need_weights=False)
            a_time = self.mha_norm(h + attn_out)
        if self.freq_attn is not None:
            f_out = self.freq_attn(h)
            a_freq = self.freq_norm(h + f_out)

        # Learnable fusion
        if self.mha is not None and self.freq_attn is not None:
            gate = torch.sigmoid(self.fusion_gate(h))     # (B, T', 1)
            h = gate * a_time + (1 - gate) * a_freq
        elif self.mha is not None:
            h = a_time
        elif self.freq_attn is not None:
            h = a_freq

        # Global average pooling over time
        z = h.mean(dim=1)                                 # (B, d_model)
        return self.head(z)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
