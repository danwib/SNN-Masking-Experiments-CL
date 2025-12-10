from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

from .config import DatasetConfig, TaskConfig


@dataclass
class TaskDataset:
    """Container for task-specific samples (flattened tensors)."""

    train_features: torch.Tensor
    train_labels: torch.Tensor
    held_out_features: torch.Tensor | None = None
    held_out_labels: torch.Tensor | None = None


SPLIT_MNIST_PAIRS: Dict[str, Tuple[int, int]] = {
    "task1": (0, 1),
    "task2": (2, 3),
    "task3": (4, 5),
    "task4": (6, 7),
    "task5": (8, 9),
    "task1_prime": (4, 5),
    "task2_prime": (6, 7),
}


def _standardize_key(raw: str) -> str:
    key = raw.lower().replace(" ", "")
    key = key.replace("'", "_prime")
    return key


def _dataset_to_tensors(dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    features = []
    labels = []
    for batch_x, batch_y in loader:
        features.append(batch_x)
        labels.append(batch_y)
    images = torch.cat(features, dim=0)
    targets = torch.cat(labels, dim=0).float()
    images = images.view(images.shape[0], -1)
    return images, targets


class _RandomDigitsDataset(Dataset):
    """Small random dataset used when MNIST is unavailable in CI."""

    def __init__(self, size: int) -> None:
        self.data = torch.rand(size, 1, 28, 28)
        self.targets = torch.randint(0, 10, (size,))

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]


class SplitMNISTLoader:
    """Loads MNIST (or FakeData fallback) and exposes digit-pair slices."""

    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg
        mnist_transform = transforms.ToTensor()
        try:
            self.train = MNIST(root=cfg.root, train=True, transform=mnist_transform, download=cfg.download)
            self.test = MNIST(root=cfg.root, train=False, transform=mnist_transform, download=cfg.download)
        except RuntimeError as exc:
            if not cfg.use_fake_data:
                raise RuntimeError(
                    "MNIST dataset not found. Set dataset.download=True or place files under "
                    f"{cfg.root}."
                ) from exc
            size = 2000
            self.train = _RandomDigitsDataset(size)
            self.test = _RandomDigitsDataset(size // 2)

        self.train_features, self.train_labels = _dataset_to_tensors(self.train)
        self.test_features, self.test_labels = _dataset_to_tensors(self.test)

    def _slice_pair(
        self, data: torch.Tensor, labels: torch.Tensor, digits: Tuple[int, int], max_samples: int | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        keep_mask = (labels == digits[0]) | (labels == digits[1])
        subset_features = data[keep_mask]
        subset_labels = labels[keep_mask]
        subset_labels = (subset_labels == digits[1]).float()

        if max_samples is not None:
            subset_features = subset_features[:max_samples]
            subset_labels = subset_labels[:max_samples]

        perm = torch.randperm(subset_features.shape[0])
        subset_features = subset_features[perm]
        subset_labels = subset_labels[perm]
        return subset_features, subset_labels

    def build_task_dataset(
        self, task: TaskConfig, max_samples: int | None, held_out_fraction: float
    ) -> TaskDataset:
        key = _standardize_key(task.dataset_variant or task.name)
        if key not in SPLIT_MNIST_PAIRS:
            raise KeyError(f"Unknown SplitMNIST variant '{task.dataset_variant}'.")
        digits = SPLIT_MNIST_PAIRS[key]
        features, labels = self._slice_pair(self.train_features, self.train_labels, digits, max_samples)

        held_out_count = int(features.shape[0] * held_out_fraction)
        if held_out_count > 0:
            return TaskDataset(
                train_features=features[held_out_count:],
                train_labels=labels[held_out_count:],
                held_out_features=features[:held_out_count],
                held_out_labels=labels[:held_out_count],
            )

        return TaskDataset(train_features=features, train_labels=labels)


def build_datasets(
    tasks: Tuple[TaskConfig, ...], dataset_cfg: DatasetConfig, seed: int
) -> Dict[str, TaskDataset]:
    """Load SplitMNIST and slice digit pairs for each task."""

    torch.manual_seed(seed)
    loader = SplitMNISTLoader(dataset_cfg)
    datasets: Dict[str, TaskDataset] = {}
    for task in tasks:
        datasets[task.name] = loader.build_task_dataset(task, dataset_cfg.max_train_samples, dataset_cfg.held_out_fraction)
    return datasets
