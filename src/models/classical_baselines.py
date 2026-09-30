"""
Classical (non-neural) baselines wrapped behind a torch-compatible interface.

We expose Random Forest, XGBoost, and a simple AutoEncoder reconstruction-error
detector. The torch wrapper lets the rest of the pipeline treat them
identically to the neural models for evaluation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def _flatten_sequences(x: np.ndarray) -> np.ndarray:
    """(N, T, F) -> (N, T*F)."""
    if x.ndim == 3:
        return x.reshape(x.shape[0], -1)
    return x


# --------------------------------------------------------------------------- #
# Random Forest                                                              #
# --------------------------------------------------------------------------- #
class RandomForestWrapper:
    """sklearn RandomForestClassifier behind a fit/predict_proba interface."""

    def __init__(self, n_estimators: int = 200, max_depth: Optional[int] = None,
                 n_jobs: int = -1, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight="balanced_subsample",
        )

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.model.fit(_flatten_sequences(x), y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_flatten_sequences(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(_flatten_sequences(x))


# --------------------------------------------------------------------------- #
# XGBoost                                                                    #
# --------------------------------------------------------------------------- #
class XGBoostWrapper:
    def __init__(self, n_estimators: int = 300, max_depth: int = 6,
                 learning_rate: float = 0.1, n_jobs: int = -1,
                 random_state: int = 42):
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError("xgboost is required: `pip install xgboost`") from e
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            random_state=random_state,
            tree_method="hist",
            eval_metric="mlogloss",
        )

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.model.fit(_flatten_sequences(x), y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_flatten_sequences(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(_flatten_sequences(x))


# --------------------------------------------------------------------------- #
# AutoEncoder anomaly baseline                                                #
# --------------------------------------------------------------------------- #
class DeepAutoEncoder(nn.Module):
    """
    Reconstruction-error baseline (Javaid et al., 2016).

    Trained on benign-only flows (or all flows in supervised mode);
    classification at test time uses a per-class reconstruction-error threshold.
    We expose it as a softmax classifier by stacking a small linear head on the
    latent representation so it plays well with the standard Trainer.
    """

    def __init__(self, input_features: int, seq_len: int,
                 latent_dim: int = 64, num_classes: int = 10):
        super().__init__()
        flat = input_features * seq_len
        self.flat = flat
        self.encoder = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, flat),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        z = self.encoder(flat)
        return self.classifier(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        return self.decoder(self.encoder(flat)).view_as(x)
