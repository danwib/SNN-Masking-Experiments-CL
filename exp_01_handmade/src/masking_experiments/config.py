from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class TaskConfig:
    """Definition for a single task slot."""

    name: str
    prompt: str
    column_index: int
    held_out: bool = False
    dataset_variant: str = "default"
    residual_from: str | None = None


@dataclass
class DatasetConfig:
    """SplitMNIST dataset options."""

    root: str = "assets/data"
    held_out_fraction: float = 0.1
    max_train_samples: int | None = None
    use_fake_data: bool = False
    download: bool = True


@dataclass
class TrainingConfig:
    """Training hyper-parameters shared by experiments."""

    epochs: int = 30
    batch_size: int = 64
    base_learning_rate: float = 0.01
    seed: int = 13
    task_schedule: List[str] = field(default_factory=list)


@dataclass
class MaskingConfig:
    """How masks affect neuron dynamics."""

    learning_rate_scale: float = 0.4
    threshold_shift: float = 0.15
    prompt_vector_dim: int = 16


@dataclass
class ModelConfig:
    """Architectural hyper-parameters for the shared SNN."""

    hidden_per_column: int = 64
    time_steps: int = 5
    beta: float = 0.9


@dataclass
class ExperimentConfig:
    """Top-level configuration for an experiment run."""

    name: str
    experiment_type: str
    tasks: List[TaskConfig] = field(default_factory=list)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def _load_task(raw: Dict[str, Any]) -> TaskConfig:
    return TaskConfig(
        name=raw["name"],
        prompt=raw.get("prompt", raw["name"]),
        column_index=raw["column_index"],
        held_out=raw.get("held_out", False),
        dataset_variant=raw.get("dataset_variant", "default"),
        residual_from=raw.get("residual_from"),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file."""

    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    tasks = [_load_task(entry) for entry in raw.get("tasks", [])]

    dataset = DatasetConfig(**raw.get("dataset", {}))
    training = TrainingConfig(**raw.get("training", {}))
    masking = MaskingConfig(**raw.get("masking", {}))
    model = ModelConfig(**raw.get("model", {}))

    return ExperimentConfig(
        name=raw.get("name", cfg_path.stem),
        experiment_type=raw["experiment_type"],
        tasks=tasks,
        dataset=dataset,
        training=training,
        masking=masking,
        model=model,
    )
