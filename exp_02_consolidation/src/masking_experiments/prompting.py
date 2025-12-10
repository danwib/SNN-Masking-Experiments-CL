from __future__ import annotations

import hashlib

import numpy as np
import torch


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
