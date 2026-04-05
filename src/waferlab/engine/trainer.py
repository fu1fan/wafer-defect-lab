"""Training loop for wafer-level classification."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.transforms import prepare_input, DEFAULT_NORM_SCALE
from ..models.resnet import WaferClassifier
from ..registry import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY


class Trainer:
    """Encapsulated training state for a wafer classifier.

    Parameters
    ----------
    model : WaferClassifier
    train_loader : DataLoader
    val_loader : DataLoader or None
    config : dict
        Training hyper-parameters (lr, epochs, weight_decay, etc.).
    device : str
    output_dir : Path
        Where to save checkpoints and logs.
    task_mode : str
        ``"binary"`` or ``"multiclass"`` – determines how labels are read
        from each batch dict.
    """

    def __init__(
        self,
        model: WaferClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: dict[str, Any],
        *,
        device: str = "cuda",
        output_dir: Path = Path("outputs"),
        task_mode: str = "binary",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_mode = task_mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs: int = int(config.get("epochs", 30))
        lr: float = float(config.get("lr", 1e-3))
        wd: float = float(config.get("weight_decay", 1e-4))
        self.grad_clip: float = float(config.get("grad_clip", 0.0))
        self.log_interval: int = int(config.get("log_interval", 50))
        self.norm_scale: float = float(config.get("norm_scale", DEFAULT_NORM_SCALE))

        # Build optimizer via registry (default: adamw).
        optimizer_name = str(config.get("optimizer", "adamw")).lower()
        opt_cfg = {
            "params": self.model.parameters(),
            "lr": lr,
            "weight_decay": wd,
            **config.get("optimizer_args", {}),
        }
        self.optimizer = OPTIMIZER_REGISTRY.build(optimizer_name, opt_cfg)

        # Build scheduler via registry (default: cosine).
        scheduler_name = str(config.get("scheduler", "cosine")).lower()
        sched_cfg = {
            "optimizer": self.optimizer,
            "epochs": self.epochs,
            **config.get("scheduler_args", {}),
        }
        self.scheduler = SCHEDULER_REGISTRY.build(scheduler_name, sched_cfg)

        # Loss function: CE (default) or Focal Loss.
        loss_type = str(config.get("loss_type", "ce")).lower()
        class_weights = config.get("class_weights")

        if loss_type == "focal":
            from .losses import FocalLoss
            focal_gamma = float(config.get("focal_gamma", 2.0))
            focal_alpha = class_weights  # reuse class_weights as alpha
            self.criterion = FocalLoss(
                gamma=focal_gamma, alpha=focal_alpha, reduction="mean",
            )
        else:
            if class_weights is not None:
                w = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
                self.criterion = nn.CrossEntropyLoss(weight=w)
            else:
                self.criterion = nn.CrossEntropyLoss()

        self.best_val_acc: float = 0.0
        self.history: list[dict[str, Any]] = []
        self.start_epoch: int = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> list[dict[str, Any]]:
        """Run the full training loop. Returns epoch-level history."""
        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self._validate() if self.val_loader else (0.0, 0.0)
            if self.scheduler is not None:
                self.scheduler.step()

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": self.optimizer.param_groups[0]["lr"],
                "elapsed": time.time() - t0,
            }
            self.history.append(record)
            self._save_history()
            print(
                f"Epoch {epoch}/{self.epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                f"lr={record['lr']:.2e}  "
                f"time={record['elapsed']:.1f}s"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint("best.pt")

            # Persist each completed epoch so interrupted runs can resume.
            self.save_checkpoint("last.pt")

        return self.history

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Restore model, optimizer, and training history from a checkpoint."""
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        self.best_val_acc = float(checkpoint.get("best_val_acc", 0.0))
        self.history = list(checkpoint.get("history", []))

        last_epoch = int(self.history[-1]["epoch"]) if self.history else 0
        self.start_epoch = last_epoch + 1

        completed_steps = min(last_epoch, self.epochs)
        if completed_steps > 0 and self.scheduler is not None:
            self.scheduler.last_epoch = completed_steps - 1

        return checkpoint

    def save_checkpoint(self, filename: str) -> Path:
        path = self.output_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_acc": self.best_val_acc,
                "history": self.history,
            },
            path,
        )
        return path

    def _save_history(self) -> Path:
        path = self.output_dir / "history.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        return path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_labels(self, batch: dict[str, Any]) -> torch.Tensor:
        """Get classification target from a data batch."""
        if self.task_mode == "binary":
            return batch["label"].to(self.device, dtype=torch.long)
        # multiclass: failure_type_idx pre-computed by the dataset/collate.
        return batch["failure_type_idx"].to(self.device, dtype=torch.long)

    def _prepare_input(self, batch: dict[str, Any]) -> torch.Tensor:
        return prepare_input(
            batch,
            device=self.device,
            target_channels=self.model.in_channels,
            norm_scale=self.norm_scale,
        )

    def _train_one_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for step, batch in enumerate(self.train_loader, 1):
            x = self._prepare_input(batch)
            y = self._extract_labels(batch)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

            if step % self.log_interval == 0:
                running_loss = total_loss / total
                running_acc = correct / total
                print(
                    f"  [epoch {epoch} step {step}] "
                    f"loss={running_loss:.4f}  acc={running_acc:.4f}"
                )

        return total_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.val_loader:  # type: ignore[union-attr]
            x = self._prepare_input(batch)
            y = self._extract_labels(batch)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)
