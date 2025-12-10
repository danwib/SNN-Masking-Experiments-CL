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
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.num_columns = num_columns
        self.hidden_per_column = model_cfg.hidden_per_column
        self.hidden_dim = self.hidden_per_column * num_columns
        self.time_steps = model_cfg.time_steps
        self.beta = model_cfg.beta
        self.base_learning_rate = base_learning_rate

        self.encoder = nn.Linear(input_dim + prompt_dim, self.hidden_dim)
        self.base_thresholds = nn.Parameter(torch.ones(self.hidden_dim))
        self.readout = nn.Linear(self.hidden_dim, num_columns)

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
        logits = self.readout(pooled)
        return logits

    def train_batch(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        target_column: int,
        mask: Mask,
    ) -> float:
        self.train()
        outputs = self.forward(inputs, mask)
        logits = outputs[:, target_column]
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()

        self._apply_updates(mask.learning_rate_scale.to(inputs.device))
        self.zero_grad(set_to_none=True)
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

        if self.readout.weight.grad is not None:
            row_scale = lr_scales.unsqueeze(1)
            self.readout.weight.data -= self.base_learning_rate * row_scale * self.readout.weight.grad
        if self.readout.bias.grad is not None:
            self.readout.bias.data -= self.base_learning_rate * lr_scales * self.readout.bias.grad

    @torch.no_grad()
    def accuracy(self, inputs: torch.Tensor, labels: torch.Tensor, mask: Mask, column_index: int) -> float:
        self.eval()
        outputs = self.forward(inputs, mask)
        logits = outputs[:, column_index]
        predictions = torch.sigmoid(logits) >= 0.5
        return float((predictions.float() == labels).float().mean().item())
