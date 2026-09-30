"""
Transformer-IDS baseline.

A 4-layer Transformer encoder over network-flow sequences, used as a strong
neural baseline. Inspired by Zheng et al. (2022) and Lin et al. (2022)
adapted to the same input format as the other baselines so it can be
benchmarked head-to-head.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerIDS(nn.Module):
    def __init__(
        self,
        input_features: int,
        seq_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        fc_dims: List[int] = (128,),
        fc_dropout: float = 0.4,
        num_classes: int = 10,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_features, d_model)
        self.pos_enc = _PositionalEncoding(d_model, max_len=seq_len * 4)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        head, prev = [], d_model
        for h in fc_dims:
            head += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(fc_dropout)]
            prev = h
        head.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.input_proj(x)            # (B, T, d_model)
        h = self.pos_enc(h)
        h = self.encoder(h)               # (B, T, d_model)
        z = h.mean(dim=1)                 # global average pool
        return self.head(z)
