from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig
from .masking import Mask


class SparseSNN(nn.Module):
    """Single shared SNN core with per-column masking overlays."""

    def __init__(
        self,
        input_dim: int,
        prompt_dim: int,
        num_columns: int,
        model_cfg: ModelConfig,
        base_learning_rate: float,
        seed: int,
        mask_mode: str = "hard_columns",
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.num_columns = num_columns
        self.hidden_per_column = model_cfg.hidden_per_column
        self.hidden_dim = self.hidden_per_column * num_columns
        self.time_steps = model_cfg.time_steps
        self.beta = model_cfg.beta
        self.base_learning_rate = base_learning_rate
        self.mask_mode = mask_mode
        self.use_soft_columns = mask_mode == "soft_columns"

        self.encoder = nn.Linear(input_dim + prompt_dim, self.hidden_dim)
        self.base_thresholds = nn.Parameter(torch.ones(self.hidden_dim))
        self.readout = nn.Linear(self.hidden_dim, num_columns)
        self.soft_readout = nn.Linear(self.hidden_per_column, 1) if self.use_soft_columns else None

    def _expand_per_column(self, values: torch.Tensor) -> torch.Tensor:
        return values.repeat_interleave(self.hidden_per_column)

    def forward(self, inputs: torch.Tensor, mask: Mask) -> torch.Tensor:
        batch_size = inputs.shape[0]
        prompt = mask.prompt_vector.unsqueeze(0).expand(batch_size, -1)
        combined = torch.cat([inputs, prompt], dim=1)

        mem = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        spikes = torch.zeros_like(mem)

        threshold_shift = self._expand_per_column(mask.threshold_shift.to(inputs.device))
        effective_thresholds = self.base_thresholds.to(inputs.device) + threshold_shift

        encoded = self.encoder(combined)
        for _ in range(self.time_steps):
            mem = self.beta * mem + encoded
            spk = torch.sigmoid(mem - effective_thresholds)
            spikes += spk
            mem = mem * (1 - spk)

        pooled = spikes / self.time_steps
        if self.use_soft_columns and mask.column_mask is not None:
            return self._forward_soft(pooled, mask)
        logits = self.readout(pooled)
        return logits

    def _forward_soft(self, pooled: torch.Tensor, mask: Mask) -> torch.Tensor:
        assert self.soft_readout is not None, "soft_readout is not initialized"
        column_mask = mask.column_mask.to(pooled.device)
        batch_size = pooled.shape[0]
        reshaped = pooled.view(batch_size, self.num_columns, self.hidden_per_column)
        mixed = torch.einsum("bk,bkh->bh", column_mask.view(1, -1), reshaped)
        logits = self.soft_readout(mixed)
        return logits.squeeze(-1)

    def _select_logits(self, outputs: torch.Tensor, mask: Mask, column_index: int) -> torch.Tensor:
        if self.use_soft_columns and mask.column_mask is not None:
            return outputs.view(-1)
        return outputs[:, column_index]

    def train_batch(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        target_column: int,
        mask: Mask,
        task_name: str | None = None,
        mask_controller=None,
        regularization: torch.Tensor | None = None,
    ) -> float:
        self.train()
        outputs = self.forward(inputs, mask)
        logits = self._select_logits(outputs, mask, target_column)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        if regularization is not None:
            loss = loss + regularization
        loss.backward()

        lr_scales = mask.learning_rate_scale.to(inputs.device).detach()
        self._apply_updates(lr_scales)
        if mask_controller is not None and task_name is not None:
            mask_controller.apply_gradients(task_name)
        self.zero_grad(set_to_none=True)
        if mask_controller is not None:
            mask_controller.zero_grad(task_name)
        return float(loss.item())

    def _apply_updates(self, lr_scales: torch.Tensor) -> None:
        hidden_scales = self._expand_per_column(lr_scales).to(self.base_thresholds.device)

        if self.encoder.weight.grad is not None:
            scale = hidden_scales.unsqueeze(1)
            self.encoder.weight.data -= self.base_learning_rate * scale * self.encoder.weight.grad
        if self.encoder.bias.grad is not None:
            self.encoder.bias.data -= self.base_learning_rate * hidden_scales * self.encoder.bias.grad

        if self.base_thresholds.grad is not None:
            self.base_thresholds.data -= self.base_learning_rate * hidden_scales * self.base_thresholds.grad

        if self.use_soft_columns:
            if self.soft_readout and self.soft_readout.weight.grad is not None:
                self.soft_readout.weight.data -= self.base_learning_rate * self.soft_readout.weight.grad
            if self.soft_readout and self.soft_readout.bias.grad is not None:
                self.soft_readout.bias.data -= self.base_learning_rate * self.soft_readout.bias.grad
        else:
            if self.readout.weight.grad is not None:
                row_scale = lr_scales.unsqueeze(1)
                self.readout.weight.data -= self.base_learning_rate * row_scale * self.readout.weight.grad
            if self.readout.bias.grad is not None:
                self.readout.bias.data -= self.base_learning_rate * lr_scales * self.readout.bias.grad

    @torch.no_grad()
    def accuracy(self, inputs: torch.Tensor, labels: torch.Tensor, mask: Mask, column_index: int) -> float:
        self.eval()
        outputs = self.forward(inputs, mask)
        logits = self._select_logits(outputs, mask, column_index)
        predictions = torch.sigmoid(logits) >= 0.5
        return float((predictions.float() == labels).float().mean().item())
