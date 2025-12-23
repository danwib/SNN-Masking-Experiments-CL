from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F

from .config import ExperimentConfig, SleepDynamicDistillationConfig, SleepPhaseConfig, TaskConfig
from .data import TaskDataset, build_datasets
from .masking import Mask, MaskController, SoftColumnMaskController
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
    mask_logs: Dict[str, Dict[str, Any]] | None = None
    metric_deltas: Dict[str, float] | None = None


class _DynamicReplayState:
    """Tracks per-sample losses and warmup progress for adaptive sleep distillation."""

    def __init__(self, dataset_size: int, cfg: SleepDynamicDistillationConfig) -> None:
        if dataset_size <= 0:
            raise ValueError("Dynamic replay requires at least one sample.")
        self.losses = torch.full((dataset_size,), cfg.initial_loss, dtype=torch.float32)
        warmup = max(0.0, cfg.warmup_passes)
        self._full_passes_remaining = int(math.floor(warmup))
        fractional = warmup - float(self._full_passes_remaining)
        fraction_count = int(math.ceil(fractional * dataset_size))
        if fractional > 0.0 and fraction_count == 0:
            fraction_count = 1
        self._fraction_indices: List[int] = []
        self._fraction_pending = fraction_count
        self._current_full_indices: List[int] = []
        self.use_moving_average = cfg.use_moving_average
        self.momentum = cfg.momentum
        self.min_probability = max(0.0, min(1.0, cfg.min_probability))
        self.probability_bias = 0.0

    def has_warmup(self) -> bool:
        return (
            self._full_passes_remaining > 0
            or bool(self._current_full_indices)
            or self._fraction_pending > 0
            or bool(self._fraction_indices)
        )

    def next_warmup_batch(self, batch_size: int, generator: torch.Generator) -> torch.Tensor | None:
        if batch_size <= 0:
            return None
        if not self._current_full_indices and self._full_passes_remaining > 0:
            perm = torch.randperm(self.losses.shape[0], generator=generator)
            self._current_full_indices = perm.tolist()
            self._full_passes_remaining -= 1
        if self._current_full_indices:
            return self._pop_indices(self._current_full_indices, batch_size)
        if self._fraction_pending > 0 and not self._fraction_indices:
            perm = torch.randperm(self.losses.shape[0], generator=generator)
            take = min(self._fraction_pending, self.losses.shape[0])
            self._fraction_indices = perm[:take].tolist()
            self._fraction_pending = 0
        if self._fraction_indices:
            return self._pop_indices(self._fraction_indices, batch_size)
        return None

    def plan_dynamic_batches(
        self, batch_size: int, cfg: SleepDynamicDistillationConfig, generator: torch.Generator
    ) -> List[torch.Tensor]:
        total = self.losses.shape[0]
        if total == 0:
            return []
        batches: List[torch.Tensor] = []
        top_ratio = max(0.0, min(1.0, cfg.top_percent))
        extra_ratio = max(0.0, min(1.0, cfg.extra_percent))
        top_k = int(round(top_ratio * total))
        if top_ratio > 0.0 and top_k == 0:
            top_k = 1
        top_k = min(top_k, total)
        extra_k = int(round(extra_ratio * total))
        available_for_extra = total - top_k
        if extra_ratio > 0.0 and extra_k == 0 and available_for_extra > 0:
            extra_k = 1
        extra_k = min(extra_k, available_for_extra)

        bias_factor = max(0.0, 1.0 + self.probability_bias)
        effective_losses = self.losses * bias_factor
        indices = torch.arange(total)
        selected: List[int] = []
        if top_k > 0:
            _, top_idx = torch.topk(effective_losses, k=top_k)
            selected.extend(top_idx.tolist())
            mask = torch.ones(total, dtype=torch.bool)
            mask[top_idx] = False
            remaining = indices[mask]
        else:
            remaining = indices

        if extra_k > 0 and remaining.numel() > 0:
            weights = effective_losses[remaining].clone()
            if torch.all(weights <= 0):
                weights = torch.ones_like(weights)
            probs = weights / weights.sum()
            if self.min_probability > 0 and remaining.numel() > 0:
                uniform = torch.full_like(probs, 1.0 / remaining.numel())
                probs = (1.0 - self.min_probability) * probs + self.min_probability * uniform
                probs = probs / probs.sum()
            sampled = torch.multinomial(probs, num_samples=extra_k, replacement=False, generator=generator)
            selected.extend(remaining[sampled].tolist())

        if not selected:
            return []

        order = torch.tensor(selected, dtype=torch.long)
        shuffle_perm = torch.randperm(order.shape[0], generator=generator)
        order = order[shuffle_perm]
        batches = [
            order[start : start + batch_size]
            for start in range(0, order.shape[0], batch_size)
            if batch_size > 0
        ]
        return batches

    def update_losses(self, indices: torch.Tensor, new_losses: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        indices = indices.view(-1)
        new_losses = new_losses.view(-1).to(self.losses.device)
        if self.use_moving_average:
            prev = self.losses[indices]
            updated = self.momentum * prev + (1.0 - self.momentum) * new_losses
            self.losses[indices] = updated
        else:
            self.losses[indices] = new_losses

    def adjust_probability_bias(self, delta: float) -> None:
        self.probability_bias = float(max(-0.5, min(2.0, self.probability_bias + delta)))

    def mean_loss(self) -> float:
        return float(self.losses.mean().item())

    @staticmethod
    def _pop_indices(storage: List[int], batch_size: int) -> torch.Tensor | None:
        if not storage:
            return None
        count = min(batch_size, len(storage))
        batch = [storage.pop() for _ in range(count)]
        return torch.tensor(batch, dtype=torch.long)


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
            outputs = model.forward(features, mask)
            logits = model._select_logits(outputs, mask, target_task.column_index)
            preds = torch.sigmoid(logits) >= 0.5
            accuracy = float((preds.float() == labels).float().mean().item())
        metrics.append(TaskMetric(task_name=task.name, accuracy=accuracy, prompt_used=context_prompt))
        if return_predictions:
            predictions[task.name] = preds.detach().cpu()
    if return_predictions:
        return metrics, predictions
    return metrics


def _mask_entropy(mask: torch.Tensor) -> float:
    safe_mask = torch.clamp(mask.float(), min=1e-8)
    entropy = -(safe_mask * safe_mask.log()).sum()
    return float(entropy.item())


def _snapshot_masks(
    controller: MaskController, prompt_overrides: Dict[str, str] | None = None
) -> Dict[str, Dict[str, Any]] | None:
    masks = controller.get_masks_dict(prompt_overrides=prompt_overrides)
    if not masks:
        return None
    summary: Dict[str, Dict[str, Any]] = {}
    for name, mask in masks.items():
        mask_list = [float(value) for value in mask.detach().cpu().tolist()]
        summary[name] = {
            "mask": mask_list,
            "entropy": _mask_entropy(mask.detach().cpu()),
        }
    occupancy = controller.column_occupancy(prompt_overrides=prompt_overrides)
    if occupancy:
        summary["_occupancy"] = {bank: [float(value) for value in tensor.tolist()] for bank, tensor in occupancy.items()}
    return summary


def _compute_metric_deltas(
    previous: Dict[str, float], current: Sequence[TaskMetric]
) -> Dict[str, float] | None:
    if not previous:
        return None
    deltas: Dict[str, float] = {}
    for metric in current:
        if metric.task_name not in previous:
            continue
        deltas[metric.task_name] = metric.accuracy - previous[metric.task_name]
    return deltas or None


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
            reg = controller.regularization_loss(task.name)
            if reg is not None:
                reg = reg.to(device)
            model.train_batch(
                batch_features.to(device),
                batch_labels.to(device),
                task.column_index,
                mask,
                task_name=task.name,
                mask_controller=controller,
                regularization=reg,
            )


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
    controller: MaskController | None = None,
    task_name: str | None = None,
    regularization: torch.Tensor | None = None,
) -> tuple[float, torch.Tensor]:
    model.train()
    student_outputs = model.forward(batch_features, student_mask)
    student_logits = model._select_logits(student_outputs, student_mask, target_column)
    temperature = max(cfg.temperature, 1e-3)
    distill_scale = _distillation_scale(cfg.temperature)
    label_losses = F.binary_cross_entropy_with_logits(student_logits, batch_labels, reduction="none")

    with torch.no_grad():
        teacher_outputs = teacher.forward(batch_features, teacher_mask)
        teacher_logits = teacher._select_logits(teacher_outputs, teacher_mask, teacher_column)
    teacher_soft = torch.sigmoid(teacher_logits / temperature)
    student_soft = student_logits / temperature
    distill_losses = F.binary_cross_entropy_with_logits(student_soft, teacher_soft, reduction="none")
    combined_losses = cfg.label_weight * label_losses + cfg.distillation_weight * distill_scale * distill_losses
    combined_loss = combined_losses.mean()
    if regularization is not None:
        combined_loss = combined_loss + regularization
    combined_loss.backward()
    model._apply_updates(student_mask.learning_rate_scale.to(batch_features.device))
    if controller is not None and task_name is not None:
        controller.apply_gradients(task_name)
    model.zero_grad(set_to_none=True)
    if controller is not None:
        controller.zero_grad(task_name)
    return float(combined_loss.item()), combined_losses.detach()


def _distillation_scale(temperature: float) -> float:
    adjusted = max(temperature, 1e-3)
    return adjusted**2 if adjusted >= 1.0 else 1.0


def _run_dynamic_sleep(
    model: SparseSNN,
    teacher: SparseSNN,
    groups: Dict[str, List[TaskConfig]],
    datasets: Dict[str, TaskDataset],
    controller: MaskController,
    device: torch.device,
    sleep_cfg: SleepPhaseConfig,
    batch_size: int,
    seed: int,
) -> None:
    dynamic_cfg = sleep_cfg.dynamic
    state_cache: Dict[str, _DynamicReplayState] = {}

    def _get_state(task: TaskConfig) -> _DynamicReplayState:
        if task.name not in state_cache:
            dataset = datasets[task.name]
            state_cache[task.name] = _DynamicReplayState(dataset.train_features.shape[0], dynamic_cfg)
        return state_cache[task.name]

    def _run_indices(base_task: TaskConfig, member: TaskConfig, batch_indices: torch.Tensor) -> torch.Tensor:
        dataset = datasets[member.name]
        features = dataset.train_features.index_select(0, batch_indices)
        labels = dataset.train_labels.index_select(0, batch_indices)
        student_mask = controller.build_mask(base_task.prompt).to(device)
        teacher_mask = controller.build_mask(member.prompt).to(device)
        regularization = controller.regularization_loss(base_task.name)
        if regularization is not None:
            regularization = regularization.to(device)
        _, per_sample_losses = _sleep_train_batch(
            model=model,
            teacher=teacher,
            batch_features=features.to(device),
            batch_labels=labels.to(device),
            student_mask=student_mask,
            teacher_mask=teacher_mask,
            target_column=base_task.column_index,
            teacher_column=member.column_index,
            cfg=sleep_cfg,
            controller=controller,
            task_name=base_task.name,
            regularization=regularization,
        )
        state = _get_state(member)
        state.update_losses(batch_indices.cpu(), per_sample_losses.detach().cpu())
        return per_sample_losses.detach()

    def _replay_sequence(base_task: TaskConfig, member: TaskConfig, order: torch.Tensor) -> None:
        if order.numel() == 0:
            return
        for start in range(0, order.shape[0], batch_size):
            batch_indices = order[start : start + batch_size]
            _run_indices(base_task, member, batch_indices.long())

    # Warm-up stage
    # Prime-specific warmup: focus on x' tasks before general passes.
    prime_passes = dynamic_cfg.prime_warmup_passes
    if prime_passes > 0:
        for base_name, members in groups.items():
            base_task = members[0]
            for member in members[1:]:
                _get_state(member)
                dataset = datasets[member.name]
                total = dataset.train_features.shape[0]
                generator = torch.Generator().manual_seed(
                    seed + 1500 + base_task.column_index * 53 + member.column_index * 59
                )
                full_passes = int(math.floor(prime_passes))
                for pass_idx in range(full_passes):
                    order = torch.randperm(total, generator=generator)
                    _replay_sequence(base_task, member, order)
                fractional = prime_passes - float(full_passes)
                if fractional > 0:
                    count = max(1, int(math.ceil(total * fractional)))
                    order = torch.randperm(total, generator=generator)[:count]
                    _replay_sequence(base_task, member, order)

    warmup_cycle = 0
    for base_name, members in groups.items():
        for member in members:
            _get_state(member)
    while True:
        made_progress = False
        for base_name, members in groups.items():
            base_task = members[0]
            for member in members:
                state = _get_state(member)
                generator = torch.Generator().manual_seed(
                    seed + 1000 + warmup_cycle * 17 + base_task.column_index * 29 + member.column_index * 31
                )
                while True:
                    batch_indices = state.next_warmup_batch(batch_size, generator)
                    if batch_indices is None:
                        break
                    made_progress = True
                    _run_indices(base_task, member, batch_indices.long())
        if not made_progress:
            break
        warmup_cycle += 1

    # Adaptive epochs
    similarity = {}
    context_vectors: Dict[str, torch.Tensor] = {}

    for epoch in range(sleep_cfg.epochs):
        for base_name, members in groups.items():
            base_task = members[0]
            for member in members:
                state = _get_state(member)
                generator = torch.Generator().manual_seed(
                    seed + epoch + base_task.column_index * 37 + member.column_index * 41
                )
                batches = state.plan_dynamic_batches(batch_size, dynamic_cfg, generator)
                for batch_indices in batches:
                    _run_indices(base_task, member, batch_indices.long())


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
    _ensure_task_ids(config.tasks)
    tasks = tuple(config.tasks)
    datasets = build_datasets(tasks, config.dataset, config.training.seed)
    any_task = next(iter(datasets.values()))
    input_dim = any_task.train_features.shape[1]
    task_input_stats: Dict[str, torch.Tensor] = {}
    for task in tasks:
        data = datasets[task.name]
        task_input_stats[task.name] = data.train_features.mean(dim=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_mode = config.masking.mode.lower()
    if mask_mode == "soft_columns":
        controller = SoftColumnMaskController(
            list(tasks),
            config.masking,
            task_input_stats=task_input_stats if config.masking.include_input_in_prompt else None,
            input_dim=input_dim,
        ).to(device)
    else:
        controller = MaskController(
            list(tasks),
            config.masking,
            task_input_stats=task_input_stats if config.masking.include_input_in_prompt else None,
            input_dim=input_dim,
        ).to(device)
    model = SparseSNN(
        input_dim=input_dim,
        num_columns=controller.num_columns,
        prompt_dim=config.masking.prompt_vector_dim,
        model_cfg=config.model,
        base_learning_rate=config.training.base_learning_rate,
        seed=config.training.seed,
        mask_mode=mask_mode,
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
                reg = controller.regularization_loss(task.name)
                if reg is not None:
                    reg = reg.to(device)
                model.train_batch(
                    batch_features.to(device),
                    batch_labels.to(device),
                    task.column_index,
                    mask,
                    task_name=task.name,
                    mask_controller=controller,
                    regularization=reg,
                )

    metrics = _evaluate(model, datasets, tasks, controller, device)
    mask_logs = _snapshot_masks(controller)
    return StageResult(label="parallel_training", metrics=metrics, mask_logs=mask_logs)


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
    mask_logs = _snapshot_masks(controller)
    results.append(StageResult(label="initial", metrics=metrics, mask_logs=mask_logs))
    prev_metrics = {metric.task_name: metric.accuracy for metric in metrics}

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
        mask_logs = _snapshot_masks(controller)
        deltas = _compute_metric_deltas(prev_metrics, metrics)
        results.append(
            StageResult(label=f"after_{task_name}", metrics=metrics, mask_logs=mask_logs, metric_deltas=deltas)
        )
        prev_metrics = {metric.task_name: metric.accuracy for metric in metrics}

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
        mask_logs = _snapshot_masks(controller)
        deltas = _compute_metric_deltas(prev_metrics, before_metrics)
        results.append(StageResult(label="before_sleep", metrics=before_metrics, mask_logs=mask_logs, metric_deltas=deltas))
        prev_metrics = {metric.task_name: metric.accuracy for metric in before_metrics}
        sleep_results = _run_sleep_phase(
            model=model,
            datasets=datasets,
            tasks=tasks,
            controller=controller,
            device=device,
            config=config,
            pre_sleep_predictions=pre_sleep_predictions,
            previous_metrics=prev_metrics,
        )
        sleep_metrics_map = {metric.task_name: metric.accuracy for metric in sleep_results[-1].metrics}
        prev_metrics = sleep_metrics_map
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
    previous_metrics: Dict[str, float],
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
        if sleep_cfg.dynamic.enabled:
            _run_dynamic_sleep(
                model=model,
                teacher=teacher,
                groups=groups,
                datasets=datasets,
                controller=controller,
                device=device,
                sleep_cfg=sleep_cfg,
                batch_size=batch_size,
                seed=seed,
            )
        else:
            for epoch in range(sleep_cfg.epochs):
                for base_name, members in groups.items():
                    base_task = members[0]
                    target_column = base_task.column_index
                    group_seed = seed + epoch + base_task.column_index * 17

                    replay_plan = []
                    for member in members:
                        repeat = sleep_cfg.held_out_replay if member is not base_task else 1
                        dataset = datasets[member.name]
                        repeat_count = max(1, repeat)
                        for repeat_idx in range(repeat_count):
                            repeat_seed = group_seed + member.column_index * 997 + repeat_idx * 131
                            replay_plan.append((member, dataset, repeat_seed))

                    if replay_plan:
                        generator = torch.Generator().manual_seed(group_seed)
                        order = torch.randperm(len(replay_plan), generator=generator).tolist()
                    else:
                        order = []

                    for plan_idx in order:
                        member, dataset, repeat_seed = replay_plan[plan_idx]
                        for batch_features, batch_labels in _iterate_batches(
                            dataset.train_features, dataset.train_labels, batch_size, repeat_seed
                        ):
                            student_mask = controller.build_mask(base_task.prompt).to(device)
                            teacher_mask = controller.build_mask(member.prompt).to(device)
                            regularization = controller.regularization_loss(base_task.name)
                            if regularization is not None:
                                regularization = regularization.to(device)
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
                                controller=controller,
                                task_name=base_task.name,
                                regularization=regularization,
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

    mask_logs = _snapshot_masks(controller, prompt_overrides=prompt_overrides or None)
    deltas = _compute_metric_deltas(previous_metrics, metrics)
    return [StageResult(label="after_sleep", metrics=metrics, mask_logs=mask_logs, metric_deltas=deltas)]
def _ensure_task_ids(tasks: Sequence[TaskConfig]) -> None:
    """Assign stable task_ids if they are not provided."""

    used: Dict[int, TaskConfig] = {}
    for task in tasks:
        if task.task_id is None:
            continue
        if task.task_id in used:
            raise ValueError(f"Duplicate task_id '{task.task_id}' found for tasks '{used[task.task_id].name}' and '{task.name}'.")
        used[task.task_id] = task
    next_id = 0
    for task in tasks:
        if task.task_id is not None:
            continue
        while next_id in used:
            next_id += 1
        task.task_id = next_id
        used[next_id] = task
        next_id += 1
