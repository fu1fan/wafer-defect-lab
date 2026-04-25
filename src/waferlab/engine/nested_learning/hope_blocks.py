"""HOPE-style self-modifying block with chunked CMS online writes.

This module is a compact PyTorch adaptation of the mechanism-level pieces in
``kmccleary3301/nested_learning``:

- differentiable self-modifying read path before CMS,
- explicit stop-gradient online writes,
- CMS fast/slow levels with update-period chunking,
- surprise gating and update telemetry.

It intentionally leaves out LLM-specific streaming state, KV-cache carry, and
Hydra/fast-state plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cms import CMS
from .levels import LevelSpec


def _chunk_loss(
    prediction: torch.Tensor,
    delta_target: torch.Tensor,
    mask_f: torch.Tensor,
    *,
    reduction: str = "sum",
) -> torch.Tensor:
    """Delta-rule target-shift loss used for CMS local writes."""
    target = (prediction.detach() - delta_target).detach()
    diff_sq = (prediction - target).pow(2)
    masked = diff_sq * mask_f
    if reduction == "mean":
        return masked.sum() / mask_f.sum().clamp(min=1.0)
    if reduction == "sum":
        return masked.sum()
    raise ValueError(f"Unsupported cms_chunk_reduction={reduction}")


def _min_update_period(levels: Sequence[LevelSpec]) -> int:
    periods = [int(spec.update_period) for spec in levels if int(spec.update_period) > 0]
    return min(periods) if periods else 1


@dataclass
class _CmsBuffer:
    inputs: list[torch.Tensor] = field(default_factory=list)
    teach: list[torch.Tensor] = field(default_factory=list)
    active: list[torch.Tensor] = field(default_factory=list)
    count: int = 0


def _pop_buffer_chunk(buffer: _CmsBuffer, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if count <= 0:
        raise ValueError("count must be positive")
    inputs: list[torch.Tensor] = []
    teach: list[torch.Tensor] = []
    active: list[torch.Tensor] = []
    remaining = count
    while remaining > 0:
        chunk_len = buffer.inputs[0].size(1)
        take = min(remaining, chunk_len)
        inputs.append(buffer.inputs[0][:, :take])
        teach.append(buffer.teach[0][:, :take])
        active.append(buffer.active[0][:, :take])
        if take == chunk_len:
            buffer.inputs.pop(0)
            buffer.teach.pop(0)
            buffer.active.pop(0)
        else:
            buffer.inputs[0] = buffer.inputs[0][:, take:]
            buffer.teach[0] = buffer.teach[0][:, take:]
            buffer.active[0] = buffer.active[0][:, take:]
        remaining -= take
    return torch.cat(inputs, dim=1), torch.cat(teach, dim=1), torch.cat(active, dim=1)


class ResidualMemory(nn.Module):
    """Small residual MLP memory used by the self-modifying read path."""

    def __init__(self, dim: int, *, out_dim: int | None = None, use_skip: bool = True) -> None:
        super().__init__()
        out_dim = dim if out_dim is None else int(out_dim)
        self.out_dim = out_dim
        self.use_skip = use_skip and out_dim == dim
        self.w2 = nn.Linear(dim, dim, bias=False)
        self.w1 = nn.Linear(dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.w1(F.gelu(self.w2(x)))
        if self.use_skip:
            return x + out
        return out


@dataclass
class SelfModifyingTitansLiteConfig:
    dim: int
    eta_scale: float = 1e-3
    qk_l2_norm: bool = True
    adaptive_q: bool = False
    local_conv_window: int | None = 4
    use_skip: bool = True
    update_lr: float = 1e-3


class SelfModifyingTitansLite(nn.Module):
    """A compact self-modifying Titans read/write path for vision tokens.

    The read path mirrors the paper-facing structure: optional local depthwise
    context, fixed/adaptive query projection, normalized query, then residual
    memory. The online write is an explicit stop-gradient local update on the
    memory read path, kept separate from the outer optimizer graph.
    """

    def __init__(self, config: SelfModifyingTitansLiteConfig) -> None:
        super().__init__()
        self.config = config
        self.local_conv: nn.Conv1d | None = None
        if config.local_conv_window is not None:
            window = int(config.local_conv_window)
            self.local_conv = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=window,
                groups=config.dim,
                bias=False,
            )
        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.q_memory = ResidualMemory(config.dim, use_skip=config.use_skip)
        self.memory = ResidualMemory(config.dim, use_skip=config.use_skip)

    def _apply_local_conv(self, x: torch.Tensor) -> torch.Tensor:
        if self.local_conv is None:
            return x
        window = self.local_conv.kernel_size[0]
        x_t = x.transpose(1, 2)
        x_t = F.pad(x_t, (window - 1, 0))
        return x + self.local_conv(x_t).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._apply_local_conv(x)
        q = self.q_memory(h) if self.config.adaptive_q else self.q_proj(h)
        if self.config.qk_l2_norm:
            q = F.normalize(q, dim=-1, eps=1e-6)
        return self.memory(q)

    @torch.no_grad()
    def apply_updates_inplace(self, x: torch.Tensor) -> None:
        """Explicit online write; this mutates memory params outside autograd."""
        modules: list[nn.Module] = [self.memory]
        if self.config.adaptive_q:
            modules.append(self.q_memory)
        params = [p for module in modules for p in module.parameters() if p.requires_grad]
        if not params:
            return
        with torch.enable_grad():
            pred = self.forward(x.detach())
            target = x.detach()
            loss = F.mse_loss(pred, target)
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        for param, grad in zip(params, grads, strict=False):
            if grad is not None:
                param.data.add_(grad, alpha=-self.config.update_lr)


@dataclass
class HOPESelfModBlockConfig:
    dim: int
    cms_levels: Sequence[LevelSpec]
    cms_hidden_multiplier: int = 4
    cms_use_layernorm: bool = True
    activation: str = "gelu"
    qk_l2_norm: bool = True
    cms_flush_partial_at_end: bool = False
    selfmod_adaptive_q: bool = False
    selfmod_local_conv_window: int | None = 4
    eta_scale: float = 1e-3
    selfmod_use_skip: bool = True
    selfmod_online_updates: bool = True
    self_mod_lr: float = 1e-3
    cms_chunk_reduction: str = "sum"
    cms_online_updates: bool = True
    surprise_threshold: float | None = None


class HOPESelfModBlock(nn.Module):
    """Self-modifying Titans read path followed by chunked CMS."""

    def __init__(self, config: HOPESelfModBlockConfig) -> None:
        super().__init__()
        self.config = config
        self.last_update_stats: Dict[str, Dict[str, float]] = {}
        self.surprise_threshold = config.surprise_threshold
        self.selfmod = SelfModifyingTitansLite(
            SelfModifyingTitansLiteConfig(
                dim=config.dim,
                eta_scale=config.eta_scale,
                qk_l2_norm=config.qk_l2_norm,
                adaptive_q=config.selfmod_adaptive_q,
                local_conv_window=config.selfmod_local_conv_window,
                use_skip=config.selfmod_use_skip,
                update_lr=config.self_mod_lr,
            )
        )
        self.cms = CMS(
            dim=config.dim,
            levels=config.cms_levels,
            hidden_multiplier=config.cms_hidden_multiplier,
            activation=config.activation,
            use_layernorm=config.cms_use_layernorm,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        surprise_value: float | None = None,
        finalize_updates: bool = True,
    ) -> torch.Tensor:
        selfmod_out = self.selfmod(x)
        if teach_signal is not None and self.training and self.config.selfmod_online_updates:
            self.selfmod.apply_updates_inplace(x.detach())

        if teach_signal is not None and self.training and self.config.cms_online_updates:
            return self._cms_forward_online(
                selfmod_out,
                teach_signal,
                surprise_value,
                finalize_updates=finalize_updates,
            )

        cms_out, cms_inputs, _cms_outputs = self.cms(selfmod_out, return_intermediates=True)
        if teach_signal is not None and self.training:
            self._update_cms(cms_inputs, teach_signal, surprise_value)
        return cms_out

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self.surprise_threshold = threshold

    def pop_update_stats(self) -> Dict[str, Dict[str, float]]:
        stats = self.last_update_stats
        self.last_update_stats = {}
        return stats

    def _passes_surprise(self, surprise_value: float | None) -> bool:
        if self.surprise_threshold is None:
            return True
        if surprise_value is None:
            return False
        return surprise_value >= self.surprise_threshold

    def _cms_forward_online(
        self,
        x: torch.Tensor,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
        *,
        finalize_updates: bool = True,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        base_chunk = _min_update_period(self.config.cms_levels)
        active_mask = teach_signal.detach().abs().sum(dim=-1) > 0
        outputs: list[torch.Tensor] = []
        buffers = {spec.name: _CmsBuffer() for spec in self.config.cms_levels}
        stats = {
            spec.name: {
                "grad_norm": 0.0,
                "chunk_tokens": 0.0,
                "gate_hit": 0.0,
                "updates_applied": 0.0,
                "tokens_flushed": 0.0,
                "pending_tokens": 0.0,
            }
            for spec in self.config.cms_levels
        }

        for start in range(0, seq_len, base_chunk):
            end = min(start + base_chunk, seq_len)
            current = x[:, start:end, :]
            chunk_teach = teach_signal[:, start:end, :]
            chunk_active = active_mask[:, start:end]
            level_inputs: dict[str, torch.Tensor] = {}
            for spec in self.config.cms_levels:
                level_inputs[spec.name] = current
                current = self.cms.blocks[spec.name](current)
            outputs.append(current)

            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                buffer.inputs.append(level_inputs[level_name].detach())
                buffer.teach.append(chunk_teach.detach())
                buffer.active.append(chunk_active.detach())
                buffer.count += end - start
                update_period = int(spec.update_period)
                while update_period > 0 and buffer.count >= update_period:
                    chunk_inputs, chunk_teach_pop, chunk_active_pop = _pop_buffer_chunk(buffer, update_period)
                    buffer.count -= update_period
                    magnitude = self._update_cms_chunk(
                        level_name,
                        chunk_inputs,
                        chunk_teach_pop,
                        chunk_active_pop,
                        surprise_value,
                    )
                    if magnitude > 0:
                        stats[level_name]["grad_norm"] += magnitude
                        stats[level_name]["chunk_tokens"] += float(update_period)
                        stats[level_name]["gate_hit"] += 1.0
                        stats[level_name]["updates_applied"] += 1.0

        if self.config.cms_flush_partial_at_end and finalize_updates:
            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                remaining = int(buffer.count)
                if remaining <= 0:
                    continue
                chunk_inputs, chunk_teach_pop, chunk_active_pop = _pop_buffer_chunk(buffer, remaining)
                buffer.count -= remaining
                magnitude = self._update_cms_chunk(
                    level_name,
                    chunk_inputs,
                    chunk_teach_pop,
                    chunk_active_pop,
                    surprise_value,
                )
                if magnitude > 0:
                    stats[level_name]["grad_norm"] += magnitude
                    stats[level_name]["chunk_tokens"] += float(remaining)
                    stats[level_name]["gate_hit"] += 1.0
                    stats[level_name]["updates_applied"] += 1.0
                    stats[level_name]["tokens_flushed"] += float(remaining)

        for spec in self.config.cms_levels:
            stats[spec.name]["pending_tokens"] = float(buffers[spec.name].count)
        for level_name, payload in stats.items():
            if (
                payload["updates_applied"] <= 0
                and payload["pending_tokens"] <= 0
                and payload["tokens_flushed"] <= 0
            ):
                continue
            if surprise_value is not None:
                payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = payload
        return torch.cat(outputs, dim=1)

    def _update_cms(
        self,
        cms_inputs: dict[str, torch.Tensor],
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        teach = teach_signal.detach()
        active_mask = teach.abs().sum(dim=-1) > 0
        for spec in self.config.cms_levels:
            level_name = spec.name
            inputs = cms_inputs[level_name].detach()
            seq_len = inputs.shape[1]
            chunk_size = int(spec.update_period)
            total_norm = 0.0
            update_events = 0
            token_events = 0
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                magnitude = self._update_cms_chunk(
                    level_name,
                    inputs[:, start:end, :],
                    teach[:, start:end, :],
                    active_mask[:, start:end],
                    surprise_value,
                )
                if magnitude <= 0:
                    continue
                total_norm += magnitude
                update_events += 1
                token_events += end - start
            if update_events > 0:
                payload: Dict[str, float] = {
                    "grad_norm": total_norm,
                    "chunk_tokens": float(token_events),
                    "gate_hit": float(update_events),
                }
                if surprise_value is not None:
                    payload["surprise_value"] = surprise_value
                self.last_update_stats[f"cms.{level_name}"] = payload

    def _update_cms_chunk(
        self,
        level_name: str,
        chunk_inputs: torch.Tensor,
        chunk_teach: torch.Tensor,
        chunk_active: torch.Tensor,
        surprise_value: float | None,
    ) -> float:
        if not self._passes_surprise(surprise_value):
            self.last_update_stats[f"gate.{level_name}"] = {"gate_hit": 0.0}
            return 0.0
        if not bool(chunk_active.any()):
            return 0.0
        block = self.cms.blocks[level_name]
        params = [p for p in block.parameters() if p.requires_grad]
        if not params:
            return 0.0
        mask_f = chunk_active.unsqueeze(-1).float()
        with torch.enable_grad():
            prediction = block(chunk_inputs.detach())
            loss = _chunk_loss(
                prediction,
                chunk_teach.detach(),
                mask_f,
                reduction=self.config.cms_chunk_reduction,
            )
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        total_norm = 0.0
        for param, grad in zip(params, grads, strict=False):
            if grad is None:
                continue
            total_norm += float(grad.detach().norm().item())
            param.data.add_(grad, alpha=-self.config.self_mod_lr)
        return total_norm
