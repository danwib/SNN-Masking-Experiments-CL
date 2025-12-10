from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from .config import MaskingConfig, TaskConfig
from .prompting import PromptEncoder


@dataclass
class Mask:
    """Mask applied to the sparse SNN."""

    learning_rate_scale: torch.Tensor
    threshold_shift: torch.Tensor
    prompt_vector: torch.Tensor

    def to(self, device: torch.device) -> "Mask":
        return Mask(
            learning_rate_scale=self.learning_rate_scale.to(device),
            threshold_shift=self.threshold_shift.to(device),
            prompt_vector=self.prompt_vector.to(device),
        )


class MaskController:
    """Build masks based on task prompts."""

    def __init__(self, tasks: List[TaskConfig], config: MaskingConfig) -> None:
        self._task_by_prompt: Dict[str, TaskConfig] = {task.prompt: task for task in tasks}
        self._tasks_by_name: Dict[str, TaskConfig] = {task.name: task for task in tasks}
        self._config = config
        self._prompt_encoder = PromptEncoder(config.prompt_vector_dim)
        self.num_columns = max(task.column_index for task in tasks) + 1

    def resolve_task(self, key: str) -> TaskConfig:
        if key in self._task_by_prompt:
            return self._task_by_prompt[key]
        if key in self._tasks_by_name:
            return self._tasks_by_name[key]
        raise KeyError(f"Unknown task or prompt '{key}'.")

    def build_mask(self, context: str) -> Mask:
        task = self.resolve_task(context)
        lr_scale = torch.full((self.num_columns,), self._config.learning_rate_scale, dtype=torch.float32)
        threshold_shift = torch.full((self.num_columns,), self._config.threshold_shift, dtype=torch.float32)

        active_columns = {task.column_index}
        if task.residual_from:
            base_task = self._tasks_by_name.get(task.residual_from)
            if base_task is None:
                raise KeyError(
                    f"Task '{task.name}' references residual_from='{task.residual_from}' which was not found."
                )
            active_columns.add(base_task.column_index)

        for idx in active_columns:
            threshold_shift[idx] = 0.0

        lr_scale[task.column_index] = 1.0
        for idx in active_columns:
            if idx != task.column_index:
                lr_scale[idx] = 0.0

        prompt_vec = self._prompt_encoder.encode(task.prompt)

        return Mask(learning_rate_scale=lr_scale, threshold_shift=threshold_shift, prompt_vector=prompt_vec)
