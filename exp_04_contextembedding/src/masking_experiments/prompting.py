from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PromptEncoder:
    """Converts task labels into deterministic vectors."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def encode(self, text: str) -> torch.Tensor:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little")
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=(self.dimension,)).astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-8
        return torch.from_numpy(vec / norm)


class PromptVectorProvider(nn.Module):
    """Retrieves prompt vectors based on config-selected mode."""

    def __init__(
        self,
        dimension: int,
        mode: str = "hash",
        num_embeddings: Optional[int] = None,
        learning_rate: float = 0.01,
        include_input: bool = False,
        input_dim: int | None = None,
        input_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.mode = mode.lower()
        self.learning_rate = learning_rate
        self.include_input = include_input
        self.input_scale = input_scale
        self._encoder = PromptEncoder(dimension)
        self._embedding: nn.Embedding | None = None
        self._input_linear: nn.Linear | None = None
        if self.mode == "learned_task_id":
            if num_embeddings is None or num_embeddings <= 0:
                raise ValueError("num_embeddings must be > 0 when mode='learned_task_id'.")
            self._embedding = nn.Embedding(num_embeddings, dimension)
            nn.init.normal_(self._embedding.weight, mean=0.0, std=1.0 / float(dimension) ** 0.5)
        elif self.mode != "hash":
            raise ValueError(f"Unsupported prompt vector mode '{mode}'.")
        if self.include_input:
            if input_dim is None or input_dim <= 0:
                raise ValueError("input_dim must be provided when include_input_in_prompt is true.")
            self._input_linear = nn.Linear(input_dim, dimension, bias=False)
            nn.init.normal_(self._input_linear.weight, mean=0.0, std=1.0 / float(dimension) ** 0.5)

    def vector(self, prompt: str, task_id: int | None = None, input_features: torch.Tensor | None = None) -> torch.Tensor:
        """Return the vector backing the provided prompt/task_id pair."""

        if self.mode == "hash" or self._embedding is None:
            base_vec = self._encoder.encode(prompt)
        else:
            if task_id is None:
                raise ValueError("task_id must be provided when using learned_task_id mode.")
            if task_id < 0 or task_id >= self._embedding.num_embeddings:
                raise ValueError(f"task_id {task_id} is out of range for prompt embeddings.")
            index = torch.tensor([task_id], dtype=torch.long, device=self._embedding.weight.device)
            base_vec = self._embedding(index).squeeze(0)
        if self.include_input and self._input_linear is not None:
            if input_features is None:
                raise ValueError("input_features must be provided when include_input_in_prompt is true.")
            projected = self._input_linear(input_features.to(base_vec.device, dtype=base_vec.dtype))
            base_vec = base_vec + self.input_scale * projected
        return F.normalize(base_vec, dim=0)

    def apply_gradients(self) -> None:
        if self._embedding is None:
            return
        if self._embedding.weight.grad is None:
            return
        self._embedding.weight.data -= self.learning_rate * self._embedding.weight.grad

    def zero_grad(self) -> None:
        if self._embedding is None:
            return
        if self._embedding.weight.grad is not None:
            self._embedding.weight.grad.detach_()
            self._embedding.weight.grad.zero_()
