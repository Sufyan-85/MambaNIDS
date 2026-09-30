"""
Baseline neural models used for comparison in the paper (§6.1):
    1. Vanilla LSTM (2-layer, H=256)
    2. Vanilla GRU  (2-layer, H=256)
    3. CNN-LSTM    (CNN feature extractor + unidirectional LSTM, no attention)
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class _SeqClassifier(nn.Module):
    """Shared FC head + global average pooling so baselines stay tidy."""

    def __init__(self, in_features: int, fc_dims: List[int], num_classes: int, dropout: float):
        super().__init__()
        layers, prev = [], in_features
        for h in fc_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, seq_out: torch.Tensor) -> torch.Tensor:
        # seq_out: (B, T, H)
        return self.head(seq_out.mean(dim=1))


class VanillaLSTM(nn.Module):
    """Plain 2-layer unidirectional LSTM baseline."""

    def __init__(
        self,
        input_features: int,
        hidden: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        fc_dims: List[int] = (128,),
        fc_dropout: float = 0.4,
        num_classes: int = 5,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = _SeqClassifier(hidden, list(fc_dims), num_classes, fc_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.classifier(h)


class VanillaGRU(nn.Module):
    """Plain 2-layer unidirectional GRU baseline."""

    def __init__(
        self,
        input_features: int,
        hidden: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        fc_dims: List[int] = (128,),
        fc_dropout: float = 0.4,
        num_classes: int = 5,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = _SeqClassifier(hidden, list(fc_dims), num_classes, fc_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.classifier(h)


class CNNLSTM(nn.Module):
    """CNN feature extractor + unidirectional LSTM (no attention) baseline."""

    def __init__(
        self,
        input_features: int,
        seq_len: int,
        cnn_channels: List[int] = (64, 128, 256),
        kernel_size: int = 3,
        pool_size: int = 2,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        fc_dims: List[int] = (128,),
        fc_dropout: float = 0.4,
        num_classes: int = 5,
    ):
        super().__init__()
        blocks = []
        prev = input_features
        for out_c in cnn_channels:
            blocks += [
                nn.Conv1d(prev, out_c, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity(),
            ]
            prev = out_c
        self.cnn = nn.Sequential(*blocks)

        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = _SeqClassifier(lstm_hidden, list(fc_dims), num_classes, fc_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, F) -> (B, F, T) for Conv1d
        h = x.transpose(1, 2)
        h = self.cnn(h)                # (B, C_last, T')
        h = h.transpose(1, 2)          # (B, T', C_last)
        h, _ = self.lstm(h)            # (B, T', H)
        return self.classifier(h)
