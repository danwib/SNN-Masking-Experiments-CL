from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F

from .config import ExperimentConfig, SleepPhaseConfig, TaskConfig
from .data import TaskDataset, build_datasets
from .masking import Mask, MaskController
from .snn import SparseSNN


@dataclass
class TaskMetric:
    task_name: str
    accuracy: float
    prompt_used: str
    error_overlap: float | None = None


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
    prompt_overrides: Dict[str, str] | None = None,
    return_predictions: bool = False,
) -> List[TaskMetric] | tuple[List[TaskMetric], Dict[str, torch.Tensor]]:
    metrics: List[TaskMetric] = []
    predictions: Dict[str, torch.Tensor] = {}
    for task in tasks:
        context_prompt = prompt_overrides.get(task.name, task.prompt) if prompt_overrides else task.prompt
        target_task = controller.resolve_task(context_prompt)
        mask = controller.build_mask(context_prompt).to(device)
        dataset = datasets[task.name]
        with torch.no_grad():
            features = dataset.train_features.to(device)
            labels = dataset.train_labels.to(device)
            logits = model.forward(features, mask)[:, target_task.column_index]
            preds = torch.sigmoid(logits) >= 0.5
            accuracy = float((preds.float() == labels).float().mean().item())
        metrics.append(TaskMetric(task_name=task.name, accuracy=accuracy, prompt_used=context_prompt))
        if return_predictions:
            predictions[task.name] = preds.detach().cpu()
    if return_predictions:
        return metrics, predictions
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


def _build_consolidation_groups(tasks: Sequence[TaskConfig]) -> Dict[str, List[TaskConfig]]:
    name_to_task = {task.name: task for task in tasks}
    groups: Dict[str, List[TaskConfig]] = {}
    for task in tasks:
        if not task.consolidate_into:
            continue
        if task.consolidate_into not in name_to_task:
            raise KeyError(
                f"Task '{task.name}' references consolidate_into='{task.consolidate_into}' which was not found."
            )
        base_task = name_to_task[task.consolidate_into]
        members = groups.setdefault(base_task.name, [base_task])
        members.append(task)
    return groups


def _sleep_train_batch(
    model: SparseSNN,
    teacher: SparseSNN,
    batch_features: torch.Tensor,
    batch_labels: torch.Tensor,
    student_mask: Mask,
    teacher_mask: Mask,
    target_column: int,
    teacher_column: int,
    cfg: SleepPhaseConfig,
) -> float:
    model.train()
    student_outputs = model.forward(batch_features, student_mask)
    student_logits = student_outputs[:, target_column]
    label_loss = F.binary_cross_entropy_with_logits(student_logits, batch_labels)

    with torch.no_grad():
        teacher_outputs = teacher.forward(batch_features, teacher_mask)
        teacher_logits = teacher_outputs[:, teacher_column]
    teacher_soft = torch.sigmoid(teacher_logits / cfg.temperature)
    student_soft = student_logits / cfg.temperature
    distill_loss = F.binary_cross_entropy_with_logits(student_soft, teacher_soft)

    combined_loss = cfg.label_weight * label_loss + cfg.distillation_weight * (cfg.temperature**2) * distill_loss
    combined_loss.backward()
    model._apply_updates(student_mask.learning_rate_scale.to(batch_features.device))
    model.zero_grad(set_to_none=True)
    return float(combined_loss.item())


def _compute_error_overlap(
    before_preds: torch.Tensor | None, after_preds: torch.Tensor, labels: torch.Tensor
) -> float | None:
    if before_preds is None:
        return None
    labels_bool = labels >= 0.5
    before_errors = before_preds.bool() ^ labels_bool
    after_errors = after_preds.bool() ^ labels_bool
    before_count = int(before_errors.sum().item())
    if before_count == 0:
        return 1.0 if int(after_errors.sum().item()) == 0 else 0.0
    overlap = int((before_errors & after_errors).sum().item())
    return float(overlap / before_count)


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

    sleep_cfg = config.sleep
    if sleep_cfg and sleep_cfg.enabled:
        before_metrics, pre_sleep_predictions = _evaluate(
            model,
            datasets,
            tasks,
            controller,
            device,
            return_predictions=True,
        )
        results.append(StageResult(label="before_sleep", metrics=before_metrics))
        sleep_results = _run_sleep_phase(
            model=model,
            datasets=datasets,
            tasks=tasks,
            controller=controller,
            device=device,
            config=config,
            pre_sleep_predictions=pre_sleep_predictions,
        )
        results.extend(sleep_results)

    return results


def _run_sleep_phase(
    model: SparseSNN,
    datasets: Dict[str, TaskDataset],
    tasks: Sequence[TaskConfig],
    controller: MaskController,
    device: torch.device,
    config: ExperimentConfig,
    pre_sleep_predictions: Dict[str, torch.Tensor],
) -> List[StageResult]:
    sleep_cfg = config.sleep
    assert sleep_cfg and sleep_cfg.enabled

    groups = _build_consolidation_groups(tasks)
    prompt_overrides: Dict[str, str] = {}
    for base_name, members in groups.items():
        base_task = members[0]
        for member in members:
            prompt_overrides[member.name] = base_task.prompt

    teacher = copy.deepcopy(model)
    teacher.eval()
    batch_size = sleep_cfg.batch_size or config.training.batch_size
    seed = config.training.seed + 97

    if groups:
        for epoch in range(sleep_cfg.epochs):
            for base_name, members in groups.items():
                base_task = members[0]
                student_mask = controller.build_mask(base_task.prompt).to(device)
                target_column = base_task.column_index
                group_seed = seed + epoch + base_task.column_index * 17

                replay_plan = []
                for member in members:
                    repeat = sleep_cfg.held_out_replay if member is not base_task else 1
                    dataset = datasets[member.name]
                    teacher_mask = controller.build_mask(member.prompt).to(device)
                    repeat_count = max(1, repeat)
                    for repeat_idx in range(repeat_count):
                        repeat_seed = group_seed + member.column_index * 997 + repeat_idx * 131
                        replay_plan.append((member, dataset, teacher_mask, repeat_seed))

                if replay_plan:
                    generator = torch.Generator().manual_seed(group_seed)
                    order = torch.randperm(len(replay_plan), generator=generator).tolist()
                else:
                    order = []

                for plan_idx in order:
                    member, dataset, teacher_mask, repeat_seed = replay_plan[plan_idx]
                    for batch_features, batch_labels in _iterate_batches(
                        dataset.train_features, dataset.train_labels, batch_size, repeat_seed
                    ):
                        _sleep_train_batch(
                            model=model,
                            teacher=teacher,
                            batch_features=batch_features.to(device),
                            batch_labels=batch_labels.to(device),
                            student_mask=student_mask,
                            teacher_mask=teacher_mask,
                            target_column=target_column,
                            teacher_column=member.column_index,
                            cfg=sleep_cfg,
                        )

    metrics, post_sleep_predictions = _evaluate(
        model,
        datasets,
        tasks,
        controller,
        device,
        prompt_overrides=prompt_overrides or None,
        return_predictions=True,
    )

    for metric in metrics:
        before = pre_sleep_predictions.get(metric.task_name)
        after = post_sleep_predictions.get(metric.task_name)
        labels = datasets[metric.task_name].train_labels
        if after is not None:
            metric.error_overlap = _compute_error_overlap(before, after, labels)

    return [StageResult(label="after_sleep", metrics=metrics)]
