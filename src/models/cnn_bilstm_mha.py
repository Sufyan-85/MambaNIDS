"""
CNN-BiLSTM-MHA architecture.

Implements the model proposed in:
    "A Novel Hybrid Deep Learning Framework for Real-Time Cyberattack
     Detection on Heterogeneous IoT Networks: CNN-BiLSTM with
     Multi-Head Attention Mechanism", Section 4.

Pipeline:
    x (B, T, F)
      -> 1D-CNN stack  -> (B, T', C_last)
      -> BiLSTM stack  -> (B, T', 2H)
      -> Multi-Head Self-Attention  -> (B, T', 2H)
      -> Global Average Pool  -> (B, 2H)
      -> FC head  -> logits (B, num_classes)
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """Conv1D -> BatchNorm -> ReLU -> MaxPool, as described in §4.2."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C_in, T)
        return self.pool(self.act(self.bn(self.conv(x))))


class CNNBiLSTMMHA(nn.Module):
    """
    Hybrid CNN-BiLSTM with Multi-Head Self-Attention.

    Parameters
    ----------
    input_features : int
        F — number of features per time step.
    seq_len : int
        T — input sequence length.
    cnn_channels : list[int]
        Filters per Conv1D block. Paper default: [64, 128, 256].
    cnn_kernel_size : int
        Kernel size k. Paper default: 3.
    cnn_pool_size : int
        MaxPool window. Paper default: 2.
    lstm_hidden : int
        Hidden size per direction. Paper default: 256.
    lstm_layers : int
        Number of stacked BiLSTM layers. Paper default: 2.
    lstm_dropout : float
        Dropout between BiLSTM layers. Paper default: 0.3.
    n_heads : int
        Number of attention heads. Paper default: 8.
    head_dim : int
        Per-head key dim d_k. Paper default: 64.
    fc_dims : list[int]
        Hidden sizes for the FC head. Paper default: [256, 128].
    fc_dropout : float
        Dropout in the FC head. Paper default: 0.4.
    num_classes : int
        Output classes (paper: 11; with NSL-KDD we use 5).
    """

    def __init__(
        self,
        input_features: int,
        seq_len: int,
        cnn_channels: List[int] = (64, 128, 256),
        cnn_kernel_size: int = 3,
        cnn_pool_size: int = 2,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        n_heads: int = 8,
        head_dim: int = 64,
        fc_dims: List[int] = (256, 128),
        fc_dropout: float = 0.4,
        num_classes: int = 5,
        use_cnn: bool = True,
        use_bilstm: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.input_features = input_features
        self.seq_len = seq_len
        self.use_cnn = use_cnn
        self.use_bilstm = use_bilstm
        self.use_attention = use_attention

        # ----- 1D-CNN stack (paper §4.2) -----
        if use_cnn:
            blocks = []
            prev = input_features
            for out_c in cnn_channels:
                blocks.append(CNNBlock(prev, out_c, cnn_kernel_size, cnn_pool_size))
                prev = out_c
            self.cnn = nn.Sequential(*blocks)
            eff_len = seq_len
            for _ in cnn_channels:
                eff_len = max(1, eff_len // cnn_pool_size)
            self.eff_len = eff_len
            self.cnn_out_channels = cnn_channels[-1]
        else:
            self.cnn = nn.Identity()
            self.eff_len = seq_len
            self.cnn_out_channels = input_features

        # ----- (Bi)LSTM stack (paper §4.3) -----
        self.bilstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=use_bilstm,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.bilstm_out = (2 if use_bilstm else 1) * lstm_hidden

        # ----- Multi-Head Attention (paper §4.4) -----
        target_embed = n_heads * head_dim
        if self.bilstm_out != target_embed:
            self.attn_proj_in = nn.Linear(self.bilstm_out, target_embed)
            attn_embed = target_embed
        else:
            self.attn_proj_in = nn.Identity()
            attn_embed = self.bilstm_out

        if use_attention:
            self.mha = nn.MultiheadAttention(
                embed_dim=attn_embed, num_heads=n_heads, batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(attn_embed)
        else:
            self.mha = None
            self.attn_norm = nn.Identity()
        self.attn_embed = attn_embed

        # ----- FC classification head (paper §4.5) -----
        head_layers = []
        prev = attn_embed
        for h in fc_dims:
            head_layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(fc_dropout)]
            prev = h
        head_layers.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*head_layers)

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, F)

        Returns
        -------
        logits : (B, num_classes)
        """
        if self.use_cnn:
            # CNN expects (B, C, T). Treat features as channels.
            h = x.transpose(1, 2)          # (B, F, T)
            h = self.cnn(h)                # (B, C_last, T')
            h = h.transpose(1, 2)          # (B, T', C_last)
        else:
            h = x                          # (B, T, F)

        # (Bi)LSTM
        h, _ = self.bilstm(h)              # (B, T', H or 2H)

        # Project to attention embed dim
        h = self.attn_proj_in(h)           # (B, T', E)

        # Self-attention (skipped under ablation)
        if self.use_attention and self.mha is not None:
            attn_out, _ = self.mha(h, h, h, need_weights=False)
            h = self.attn_norm(h + attn_out)

        # Global average pooling over time
        z = h.mean(dim=1)                  # (B, E)
        return self.head(z)

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.softmax(self.forward(x), dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
