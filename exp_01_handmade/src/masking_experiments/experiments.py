from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch

from .config import ExperimentConfig, TaskConfig
from .data import TaskDataset, build_datasets
from .masking import MaskController
from .snn import SparseSNN


@dataclass
class TaskMetric:
    task_name: str
    accuracy: float


@dataclass
class StageResult:
    label: str
    metrics: List[TaskMetric]


@dataclass
class ExperimentResult:
    name: str
    stages: List[StageResult]


def _iterate_batches(
    features: torch.Tensor, labels: torch.Tensor, batch_size: int, seed: int
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(features.shape[0], generator=generator)
    for start in range(0, features.shape[0], batch_size):
        batch_idx = indices[start : start + batch_size]
        yield features[batch_idx], labels[batch_idx]


def _evaluate(
    model: SparseSNN,
    datasets: Dict[str, TaskDataset],
    tasks: Sequence[TaskConfig],
    controller: MaskController,
    device: torch.device,
) -> List[TaskMetric]:
    metrics: List[TaskMetric] = []
    for task in tasks:
        dataset = datasets[task.name]
        mask = controller.build_mask(task.prompt).to(device)
        acc = model.accuracy(
            dataset.train_features.to(device), dataset.train_labels.to(device), mask, task.column_index
        )
        metrics.append(TaskMetric(task_name=task.name, accuracy=acc))
    return metrics


def _train_task(
    model: SparseSNN,
    task: TaskConfig,
    dataset: TaskDataset,
    controller: MaskController,
    batch_size: int,
    epochs: int,
    seed: int,
    device: torch.device,
) -> None:
    for epoch in range(epochs):
        epoch_seed = seed + epoch
        for batch_features, batch_labels in _iterate_batches(dataset.train_features, dataset.train_labels, batch_size, epoch_seed):
            mask = controller.build_mask(task.prompt).to(device)
            model.train_batch(batch_features.to(device), batch_labels.to(device), task.column_index, mask)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    tasks = tuple(config.tasks)
    datasets = build_datasets(tasks, config.dataset, config.training.seed)
    controller = MaskController(list(tasks), config.masking)

    any_task = next(iter(datasets.values()))
    input_dim = any_task.train_features.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseSNN(
        input_dim=input_dim,
        num_columns=controller.num_columns,
        prompt_dim=config.masking.prompt_vector_dim,
        model_cfg=config.model,
        base_learning_rate=config.training.base_learning_rate,
        seed=config.training.seed,
    ).to(device)

    trainable_tasks = [task for task in tasks if not task.held_out]
    if not trainable_tasks:
        raise ValueError("No trainable tasks defined (all are held out).")

    if config.experiment_type.lower() == "parallel":
        stages = [_run_parallel(model, datasets, trainable_tasks, controller, config, device)]
    elif config.experiment_type.lower() == "sequential":
        stages = _run_sequential(model, datasets, trainable_tasks, controller, config, device)
    else:
        raise ValueError(f"Unsupported experiment type '{config.experiment_type}'.")

    return ExperimentResult(name=config.name, stages=stages)


def _run_parallel(
    model: SparseSNN,
    datasets: Dict[str, TaskDataset],
    tasks: Sequence[TaskConfig],
    controller: MaskController,
    config: ExperimentConfig,
    device: torch.device,
) -> StageResult:
    for epoch in range(config.training.epochs):
        for task in tasks:
            data = datasets[task.name]
            epoch_seed = config.training.seed + epoch + task.column_index
            for batch_features, batch_labels in _iterate_batches(
                data.train_features, data.train_labels, config.training.batch_size, epoch_seed
            ):
                mask = controller.build_mask(task.prompt).to(device)
                model.train_batch(batch_features.to(device), batch_labels.to(device), task.column_index, mask)

    metrics = _evaluate(model, datasets, tasks, controller, device)
    return StageResult(label="parallel_training", metrics=metrics)


def _run_sequential(
    model: SparseSNN,
    datasets: Dict[str, TaskDataset],
    tasks: Sequence[TaskConfig],
    controller: MaskController,
    config: ExperimentConfig,
    device: torch.device,
) -> List[StageResult]:
    results: List[StageResult] = []
    metrics = _evaluate(model, datasets, tasks, controller, device)
    results.append(StageResult(label="initial", metrics=metrics))

    schedule = config.training.task_schedule or [task.name for task in tasks]
    name_to_task = {task.name: task for task in tasks}

    for task_name in schedule:
        if task_name not in name_to_task:
            continue
        _train_task(
            model=model,
            task=name_to_task[task_name],
            dataset=datasets[task_name],
            controller=controller,
            batch_size=config.training.batch_size,
            epochs=config.training.epochs,
            seed=config.training.seed,
            device=device,
        )
        metrics = _evaluate(model, datasets, tasks, controller, device)
        results.append(StageResult(label=f"after_{task_name}", metrics=metrics))

    return results
