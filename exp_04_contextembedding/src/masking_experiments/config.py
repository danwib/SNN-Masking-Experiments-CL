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
    task_id: int | None = None
    held_out: bool = False
    dataset_variant: str = "default"
    dataset_name: str | None = None
    residual_from: str | None = None
    consolidate_into: str | None = None


@dataclass
class DatasetConfig:
    """Dataset options shared across MNIST/FashionMNIST pipelines."""

    root: str = "assets/data"
    default_dataset: str = "mnist"
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
    prompt_mode: str = "hash"
    prompt_learning_rate: float = 0.01
    include_input_in_prompt: bool = False
    prompt_input_scale: float = 1.0
    mode: str = "hard_columns"
    soft_columns: "SoftColumnMaskConfig" = field(default_factory=lambda: SoftColumnMaskConfig())


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
    sleep: "SleepPhaseConfig" | None = None


@dataclass
class SleepDynamicDistillationConfig:
    """Adaptive replay configuration for the sleep phase."""

    enabled: bool = False
    warmup_passes: float = 1.0
    prime_warmup_passes: float = 0.0
    top_percent: float = 0.3
    extra_percent: float = 0.1
    min_probability: float = 0.05
    use_moving_average: bool = True
    momentum: float = 0.9
    initial_loss: float = 1.0
    similarity_influence: float = 0.0
    gradient_similarity_influence: float = 0.0
    gradient_update_interval: int = 1


@dataclass
class SleepPhaseConfig:
    """Configuration for the optional sleep consolidation phase."""

    enabled: bool = False
    epochs: int = 10
    batch_size: int | None = None
    label_weight: float = 0.5
    distillation_weight: float = 0.5
    temperature: float = 2.0
    held_out_replay: int = 1
    dynamic: SleepDynamicDistillationConfig = field(default_factory=SleepDynamicDistillationConfig)


@dataclass
class SoftColumnMaskConfig:
    """Options for the learnable soft-column controller."""

    total_columns: int = 0
    base_columns: int | None = None
    base_tasks: List[str] = field(default_factory=list)
    novel_tasks: List[str] = field(default_factory=list)
    temperature: float = 0.5
    lambda_entropy: float = 0.0
    lambda_balance: float = 0.0
    lambda_overlap_novel: float = 0.0
    mask_learning_rate: float = 0.01


def _load_task(raw: Dict[str, Any]) -> TaskConfig:
    return TaskConfig(
        name=raw["name"],
        prompt=raw.get("prompt", raw["name"]),
        column_index=raw["column_index"],
        task_id=raw.get("task_id"),
        held_out=raw.get("held_out", False),
        dataset_variant=raw.get("dataset_variant", "default"),
        dataset_name=raw.get("dataset_name"),
        residual_from=raw.get("residual_from"),
        consolidate_into=raw.get("consolidate_into"),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file."""

    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    tasks = [_load_task(entry) for entry in raw.get("tasks", [])]

    dataset = DatasetConfig(**raw.get("dataset", {}))
    training = TrainingConfig(**raw.get("training", {}))

    masking_raw = raw.get("masking", {})
    soft_columns_raw = masking_raw.get("soft_columns", {})
    soft_columns_cfg = SoftColumnMaskConfig(**soft_columns_raw)
    masking_kwargs = {key: value for key, value in masking_raw.items() if key != "soft_columns"}
    masking = MaskingConfig(**masking_kwargs, soft_columns=soft_columns_cfg)

    model = ModelConfig(**raw.get("model", {}))
    sleep_cfg_raw = raw.get("sleep") or {}
    if sleep_cfg_raw:
        dynamic_raw = sleep_cfg_raw.get("dynamic", {}) or {}
        sleep_kwargs = {key: value for key, value in sleep_cfg_raw.items() if key != "dynamic"}
        sleep = SleepPhaseConfig(**sleep_kwargs, dynamic=SleepDynamicDistillationConfig(**dynamic_raw))
    else:
        sleep = SleepPhaseConfig()

    return ExperimentConfig(
        name=raw.get("name", cfg_path.stem),
        experiment_type=raw["experiment_type"],
        tasks=tasks,
        dataset=dataset,
        training=training,
        masking=masking,
        model=model,
        sleep=sleep,
    )
