"""
Training loop with Adam + cosine-annealing LR + early stopping.
Mirrors the protocol in §4.5 / §6.1 of the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class TrainingHistory:
    train_loss: list
    val_loss: list
    val_acc: list
    best_epoch: int
    best_val_loss: float


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


class Trainer:
    """Generic trainer for any of our models — they all return logits."""

    def __init__(
        self,
        model: nn.Module,
        train_cfg,
        ckpt_dir: Path,
        run_name: str = "run",
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.cfg = train_cfg
        self.device = _resolve_device(train_cfg.device)
        self.model = model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device)
                                             if class_weights is not None else None)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            betas=(train_cfg.beta1, train_cfg.beta2),
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=train_cfg.epochs)

        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.ckpt_dir / f"{run_name}_best.pt"
        self.run_name = run_name

    # ------------------------------------------------------------------ #
    # epoch loops                                                        #
    # ------------------------------------------------------------------ #
    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            logits = self.model(xb)
            loss = self.criterion(logits, yb)
            loss.backward()
            if self.cfg.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        return total / max(n, 1)

    @torch.no_grad()
    def _eval(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
            logits = self.model(xb)
            loss = self.criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=-1) == yb).sum().item()
            n += xb.size(0)
        return total_loss / max(n, 1), correct / max(n, 1)

    # ------------------------------------------------------------------ #
    # public                                                             #
    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingHistory:
        best_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        hist: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_loss, val_acc = self._eval(val_loader)
            self.scheduler.step()

            hist["train_loss"].append(train_loss)
            hist["val_loss"].append(val_loss)
            hist["val_acc"].append(val_acc)

            tqdm.write(
                f"[{self.run_name}] epoch {epoch:3d}/{self.cfg.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_acc*100:.2f}%"
            )

            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(self.model.state_dict(), self.ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stop_patience:
                    tqdm.write(f"[{self.run_name}] early stopping at epoch {epoch}")
                    break

        # Restore best weights
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        return TrainingHistory(
            train_loss=hist["train_loss"],
            val_loss=hist["val_loss"],
            val_acc=hist["val_acc"],
            best_epoch=best_epoch,
            best_val_loss=best_loss,
        )

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_y, all_pred = [], []
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            preds = self.model(xb).argmax(dim=-1).cpu().numpy()
            all_pred.append(preds)
            all_y.append(yb.numpy())
        return np.concatenate(all_y), np.concatenate(all_pred)
