"""Training loop for wafer-level classification."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.transforms import prepare_input, DEFAULT_NORM_SCALE
from ..models.resnet import WaferClassifier
from ..registry import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
from .prototype_memory import PrototypeMemory


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

        # Knowledge distillation (LwF / DER style).
        self.teacher_model: nn.Module | None = None
        self.kd_alpha: float = float(config.get("kd_alpha", 1.0))
        self.kd_temperature: float = float(config.get("kd_temperature", 2.0))
        # Additive KD: loss = CE + kd_lambda * KD (standard LwF formulation).
        # When kd_lambda > 0, additive mode is used (kd_alpha is ignored).
        self.kd_lambda: float = float(config.get("kd_lambda", 0.0))
        # Old-class indices to mask KD loss (None = use all logits).
        self.kd_old_classes: list[int] | None = None
        # Only apply KD to old-class (replay) samples, not new-class ones.
        self.kd_replay_only: bool = bool(config.get("kd_replay_only", False))

        # Nested-learning teach-signal mechanism.
        nested_cfg = config.get("nested_teach", {})
        self.nested_teach_enabled: bool = bool(nested_cfg.get("enabled", False))
        self.nested_teach_scale: float = float(nested_cfg.get("teach_scale", 1.0))
        self.nested_teach_warmup: int = int(nested_cfg.get("warmup_epochs", 2))
        if self.nested_teach_enabled:
            print(
                f"[nested_teach] enabled  scale={self.nested_teach_scale}  "
                f"warmup={self.nested_teach_warmup} epochs"
            )

        # Optional prototype memory (Nested-Learning-inspired).
        proto_cfg = config.get("prototype")
        if proto_cfg and proto_cfg.get("enabled", False):
            feat_dim = self._infer_feat_dim()
            self.prototype: PrototypeMemory | None = PrototypeMemory(
                feat_dim=feat_dim,
                num_classes=model.num_classes,
                momentum=float(proto_cfg.get("momentum", 0.99)),
                surprise_threshold=float(proto_cfg.get("surprise_threshold", 0.0)),
                aux_weight=float(proto_cfg.get("aux_weight", 0.1)),
                warmup_epochs=int(proto_cfg.get("warmup_epochs", 3)),
            )
            print(
                f"[prototype] enabled  feat_dim={feat_dim}  "
                f"momentum={self.prototype.momentum}  "
                f"surprise={self.prototype.surprise_threshold}  "
                f"aux_weight={self.prototype.aux_weight}  "
                f"warmup={self.prototype.warmup_epochs} epochs"
            )
        else:
            self.prototype = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_teacher(self, teacher_model: nn.Module | None) -> None:
        """Set a frozen teacher model for knowledge distillation (LwF)."""
        if teacher_model is not None:
            teacher_model = teacher_model.to(self.device)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)
        self.teacher_model = teacher_model

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

    def load_backbone(self, path: str | Path) -> None:
        """Load backbone weights from a checkpoint, skipping parameters whose shapes
        differ between checkpoint and current model (e.g. the classification head).

        Intended for cross-task transfer such as multiclass pre-training followed by
        binary fine-tuning.  Unlike :meth:`load_checkpoint`, this method does **not**
        restore the optimizer state, training history, or ``best_val_acc`` -- the
        caller starts a fresh training run with transferred backbone weights.
        """
        checkpoint = torch.load(Path(path), map_location=self.device)
        src_state = checkpoint["model_state_dict"]
        dst_state = self.model.state_dict()

        compatible: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for key, val in src_state.items():
            if key not in dst_state:
                skipped.append(f"  {key}: not present in current model")
            elif val.shape != dst_state[key].shape:
                skipped.append(
                    f"  {key}: checkpoint shape {tuple(val.shape)} "
                    f"!= model shape {tuple(dst_state[key].shape)}"
                )
            else:
                compatible[key] = val

        self.model.load_state_dict(compatible, strict=False)

        if skipped:
            print(f"[load_backbone] skipped {len(skipped)} key(s) due to shape mismatch:")
            for msg in skipped:
                print(msg)
        print(
            f"[load_backbone] loaded {len(compatible)}/{len(src_state)} "
            f"parameter tensors from {path}"
        )

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Restore model, optimizer, and training history from a checkpoint.

        Requires an exact parameter-shape match.  Use :meth:`load_backbone` when
        transferring weights across tasks with different ``num_classes``.
        """
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

        proto_state = checkpoint.get("prototype_state")
        if proto_state is not None and self.prototype is not None:
            self.prototype.load_state_dict(proto_state)

        return checkpoint

    def save_checkpoint(self, filename: str) -> Path:
        path = self.output_dir / filename
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
        }
        if self.prototype is not None:
            payload["prototype_state"] = self.prototype.state_dict()
        torch.save(payload, path)
        return path

    def _save_history(self) -> Path:
        path = self.output_dir / "history.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        return path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _infer_feat_dim(self) -> int:
        """Derive the feature dimension from the model's classification head."""
        if hasattr(self.model, "head") and isinstance(self.model.head, nn.Sequential):
            return self.model.head[0].in_features  # first Linear layer
        if hasattr(self.model, "fc") and isinstance(self.model.fc, nn.Linear):
            return self.model.fc.in_features
        raise RuntimeError("Cannot infer feat_dim: model has no .fc or .head")

    def _classify_features(self, feat: torch.Tensor) -> torch.Tensor:
        """Compute logits from pre-head features (avoids double backbone pass)."""
        dropped = self.model.drop(feat) if hasattr(self.model, "drop") else feat
        if hasattr(self.model, "head") and isinstance(self.model.head, nn.Module):
            return self.model.head(dropped)
        return self.model.fc(dropped)

    def _extract_labels(self, batch: dict[str, Any]) -> torch.Tensor:
        """Get classification target from a data batch."""
        if self.task_mode == "binary":
            return batch["label"].to(self.device, dtype=torch.long)
        # multiclass: failure_type_idx pre-computed by the dataset/collate.
        return batch["failure_type_idx"].to(self.device, dtype=torch.long)

    def _compute_kd_loss(
        self, student_logits: torch.Tensor, x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute masked knowledge distillation loss (LwF / iCaRL style).

        If ``kd_old_classes`` is set, only the old-class logit dimensions
        are used for the KL-divergence.  This avoids distilling the
        teacher's noise on new-class neurons.

        If ``kd_replay_only`` is True and *y* is provided, the KD loss is
        computed **only** on samples whose label belongs to ``kd_old_classes``.
        This prevents the teacher's meaningless predictions on new-class
        samples from interfering with CE-driven learning.
        """
        # Per-sample masking: only distill on old-class (replay) samples.
        if self.kd_replay_only and y is not None and self.kd_old_classes is not None:
            old_set = set(self.kd_old_classes)
            mask = torch.tensor(
                [yi.item() in old_set for yi in y],
                dtype=torch.bool, device=y.device,
            )
            if mask.sum() == 0:
                return torch.tensor(0.0, device=student_logits.device)
            student_logits = student_logits[mask]
            x = x[mask]

        with torch.no_grad():
            teacher_logits = self.teacher_model(x)
        T = self.kd_temperature
        if self.kd_old_classes is not None:
            idx = self.kd_old_classes
            s = student_logits[:, idx]
            t = teacher_logits[:, idx]
        else:
            s = student_logits
            t = teacher_logits
        return F.kl_div(
            F.log_softmax(s / T, dim=1),
            F.softmax(t / T, dim=1),
            reduction="batchmean",
        ) * (T * T)

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
        use_proto_loss = (
            self.prototype is not None
            and epoch > self.prototype.warmup_epochs
        )
        use_nested_teach = (
            self.nested_teach_enabled
            and epoch > self.nested_teach_warmup
            and hasattr(self.model, "forward_with_teach")
        )

        for step, batch in enumerate(self.train_loader, 1):
            x = self._prepare_input(batch)
            y = self._extract_labels(batch)

            if use_nested_teach:
                logits, loss = self._nested_teach_step(x, y)
            elif self.prototype is not None:
                # Split forward: features → logits  (single backbone pass).
                feat = self.model.forward_features(x)
                logits = self._classify_features(feat)
                loss = self.criterion(logits, y)
            else:
                logits = self.model(x)
                loss = self.criterion(logits, y)

            # Prototype: surprise-gated update + optional auxiliary loss.
            if self.prototype is not None:
                with torch.no_grad():
                    per_sample_loss = F.cross_entropy(
                        logits.detach(), y, reduction="none",
                    )
                self.prototype.update(feat.detach(), y, per_sample_loss)

                if use_proto_loss:
                    loss = loss + self.prototype.alignment_loss(feat, y)

            # Knowledge distillation (LwF): KL-div between student and teacher.
            if self.teacher_model is not None and not use_nested_teach:
                kd_loss = self._compute_kd_loss(logits, x, y)
                if self.kd_lambda > 0:
                    loss = loss + self.kd_lambda * kd_loss
                else:
                    loss = self.kd_alpha * loss + (1.0 - self.kd_alpha) * kd_loss

            # nested_teach_step already did zero_grad / backward / step.
            if not use_nested_teach:
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

    def _nested_teach_step(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Two-pass nested-learning training step.

        Pass 1 (standard): forward → loss → backward → optimizer step.
        Pass 2 (teach):    compute teach signal from loss gradient w.r.t.
                           token features, then run forward_with_teach to
                           trigger inner CMS updates in the nested blocks.
        """
        # Pass 1: standard forward + backward.
        # We need token-level features to compute the teach signal.
        feat = self.model.stem(x)
        feat = self.model.patch_embed(feat)
        B, D, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # [B, T, D]
        tokens_for_grad = tokens.detach().requires_grad_(True)

        # Forward through nested blocks (no teach signal).
        current = tokens_for_grad
        for block in self.model.nested_blocks:
            current = block(current)
        pooled = current.mean(dim=1)
        pooled = self.model.norm(pooled)
        logits = self.model.fc(self.model.drop(pooled))
        loss = self.criterion(logits, y)

        # Compute teach signal: gradient of loss w.r.t. token features.
        teach_signal = torch.autograd.grad(
            loss, tokens_for_grad, retain_graph=False, create_graph=False,
        )[0]
        teach_signal = teach_signal.detach() * self.nested_teach_scale

        # Surprise value: mean norm of the teach signal.
        surprise_value = float(teach_signal.norm(dim=-1).mean().item())

        # Standard backward on the full model.
        self.optimizer.zero_grad()
        logits_full = self.model(x)
        loss_full = self.criterion(logits_full, y)

        # Knowledge distillation in nested-teach path.
        if self.teacher_model is not None:
            kd_loss = self._compute_kd_loss(logits_full, x, y)
            if self.kd_lambda > 0:
                loss_full = loss_full + self.kd_lambda * kd_loss
            else:
                loss_full = self.kd_alpha * loss_full + (1.0 - self.kd_alpha) * kd_loss

        loss_full.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        # Pass 2: inner CMS updates via teach signal.
        # The NestedBlock._update_cms handles its own grad context internally
        # via torch.enable_grad(), so we don't wrap in torch.no_grad().
        self.model.train()  # nested blocks check .training for updates
        self.model.forward_with_teach(
            x.detach(),
            teach_signal=teach_signal,
            surprise_value=surprise_value,
        )

        return logits_full.detach(), loss_full.detach()

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
