"""
Central configuration for CNN-BiLSTM-MHA experiments.

Hyperparameters mirror Section 4 of the paper:
"A Novel Hybrid Deep Learning Framework for Real-Time Cyberattack
 Detection on Heterogeneous IoT Networks: CNN-BiLSTM with
 Multi-Head Attention Mechanism"
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass
class DataConfig:
    # NSL-KDD has 41 features after one-hot encoding it expands to ~122.
    # Paper uses 62 features (after MIFS reduction from 78). We mirror the
    # filtering threshold and let the preprocessor decide the final count.
    dataset: str = "nsl-kdd"
    sequence_length: int = 10           # T = 10 (paper §3.3)
    mifs_threshold: float = 0.05        # MI >= 0.05 (paper §3.3)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    use_smote: bool = True              # SMOTE on training split only
    random_state: int = 42


@dataclass
class ModelConfig:
    # 1D-CNN (paper §4.2)
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_size: int = 3
    cnn_pool_size: int = 2

    # BiLSTM (paper §4.3)
    lstm_hidden: int = 256              # H = 256 per direction
    lstm_layers: int = 2
    lstm_dropout: float = 0.3

    # Multi-Head Attention (paper §4.4)
    n_heads: int = 8                    # n_h = 8
    head_dim: int = 64                  # d_k = 64

    # Classification head (paper §4.5)
    fc_dims: List[int] = field(default_factory=lambda: [256, 128])
    fc_dropout: float = 0.4
    num_classes: int = 5                # NSL-KDD: Normal + 4 attack categories


@dataclass
class TrainConfig:
    # paper §4.5
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-4          # L2 lambda
    batch_size: int = 512
    epochs: int = 100
    early_stop_patience: int = 10
    scheduler: str = "cosine"           # cosine annealing
    grad_clip: float = 1.0
    device: str = "cuda"                # falls back to cpu in trainer
    num_workers: int = 2
    log_interval: int = 50              # batches between TB writes


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 42


def default_config() -> Config:
    """Factory used by scripts/train.py and notebooks."""
    return Config()
